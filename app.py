import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import weibull_min
import io

st.set_page_config(page_title="Dashboard de Manutenção CIM", layout="wide")

# --- 1. FUNÇÃO PARA DOWNLOAD DE TEMPLATE ---
def generate_template():
    columns = [
        'MODELO EQUIPAMENTO', 'EQUIPAMENTO', 'TIPO', 
        'RESPONSABILIDADE NÍVEL 1', 'RESPONSABILIDADE NÍVEL 2', 
        'GRUPO', 'SUBGRUPO', 'DATA INÍCIO', 'DATA FIM', 
        'HORA INÍCIO', 'HORA FIM', 'TOTAL HORAS DECIMAIS'
    ]
    df_template = pd.DataFrame(columns=columns)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Planilha1')
    return output.getvalue()

st.sidebar.download_button(
    label="📥 Baixar Planilha Modelo",
    data=generate_template(),
    file_name="modelo_manutencao.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- 2. UPLOAD E LEITURA DE DADOS ---
st.title("⚙️ Dashboard de Engenharia de Manutenção e Confiabilidade")
uploaded_file = st.sidebar.file_uploader("Carregue sua planilha de ordens de serviço (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    
    # Limpeza de nomes de colunas
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    
    # Tratamento de datas e horas
    df['DATA INÍCIO'] = pd.to_datetime(df['DATA INÍCIO'], errors='coerce')
    df['DATA FIM'] = pd.to_datetime(df['DATA FIM'], errors='coerce')
    df['TOTAL HORAS DECIMAIS'] = pd.to_numeric(df['TOTAL HORAS DECIMAIS'], errors='coerce').fillna(0)
    df['TIPO'] = df['TIPO'].astype(str).str.upper().str.strip()

    # --- 3. FILTROS LATERAIS ---
    st.sidebar.header("Filtros")
    
    # Seleção de datas
    min_date = df['DATA INÍCIO'].min()
    max_date = df['DATA FIM'].max()
    date_range = st.sidebar.date_input("Período", [min_date, max_date])
    
    # Filtros categóricos
    def create_filter(col_name, label):
        if col_name in df.columns:
            options = df[col_name].dropna().unique().tolist()
            selected = st.sidebar.multiselect(label, options, default=options)
            return selected
        return []

    mod_equip = create_filter('MODELO EQUIPAMENTO', 'Modelo Equipamento')
    equip = create_filter('EQUIPAMENTO', 'Equipamento')
    tipo_os = create_filter('TIPO', 'Tipo de OS')
    resp_n1 = create_filter('RESPONSABILIDADE NÍVEL 1', 'Responsabilidade N1')
    resp_n2 = create_filter('RESPONSABILIDADE NÍVEL 2', 'Responsabilidade N2')
    grupo = create_filter('GRUPO', 'Grupo')
    subgrupo = create_filter('SUBGRUPO', 'Subgrupo')

    # Aplicando os filtros
    mask = (
        (df['MODELO EQUIPAMENTO'].isin(mod_equip)) &
        (df['EQUIPAMENTO'].isin(equip)) &
        (df['TIPO'].isin(tipo_os)) &
        (df['RESPONSABILIDADE NÍVEL 1'].isin(resp_n1)) &
        (df['RESPONSABILIDADE NÍVEL 2'].isin(resp_n2)) &
        (df['GRUPO'].isin(grupo)) &
        (df['SUBGRUPO'].isin(subgrupo))
    )
    
    if len(date_range) == 2:
        mask = mask & (df['DATA INÍCIO'].dt.date >= date_range[0]) & (df['DATA INÍCIO'].dt.date <= date_range[1])
        
    df_filtered = df[mask]

    # --- 4. KPIs BÁSICOS ---
    st.markdown("### 📊 Indicadores Principais (KPIs)")
    
    corretivas = df_filtered[df_filtered['TIPO'] == 'CORRETIVA']
    num_falhas = len(corretivas)
    total_downtime = corretivas['TOTAL HORAS DECIMAIS'].sum()
    
    mttr = total_downtime / num_falhas if num_falhas > 0 else 0
    
    # Estimativa de MTBF (assumindo 24h/dia)
    dias_operacao = (date_range[1] - date_range[0]).days if len(date_range) == 2 else 30
    dias_operacao = max(dias_operacao, 1)
    qtd_equipamentos = df_filtered['EQUIPAMENTO'].nunique()
    horas_disponiveis = dias_operacao * 24 * qtd_equipamentos
    mtbf = (horas_disponiveis - total_downtime) / num_falhas if num_falhas > 0 else 0
    
    # % Preventiva / Corretiva
    tipos_count = df_filtered['TIPO'].value_counts(normalize=True) * 100
    perc_prev = tipos_count.get('PREVENTIVA', 0)
    perc_corr = tipos_count.get('CORRETIVA', 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MTBF (Horas)", f"{mtbf:.2f}")
    col2.metric("MTTR (Horas)", f"{mttr:.2f}")
    col3.metric("% Preventiva", f"{perc_prev:.1f}%")
    col4.metric("% Corretiva", f"{perc_corr:.1f}%")

    # --- 5. PERFIL DE PERDAS (PARETO) ---
    st.markdown("### 📉 Perfil de Perdas (Análise de Pareto)")
    tab1, tab2, tab3 = st.tabs(["Por Equipamento", "Por Grupo", "Por Subgrupo"])
    
    def plot_pareto(data, col_name, title):
        df_pareto = data.groupby(col_name)['TOTAL HORAS DECIMAIS'].sum().reset_index()
        df_pareto = df_pareto.sort_values(by='TOTAL HORAS DECIMAIS', ascending=False)
        df_pareto['Porcentagem Acumulada'] = df_pareto['TOTAL HORAS DECIMAIS'].cumsum() / df_pareto['TOTAL HORAS DECIMAIS'].sum() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_pareto[col_name], y=df_pareto['TOTAL HORAS DECIMAIS'], name="Horas de Máquina Parada", marker_color='indianred'))
        fig.add_trace(go.Scatter(x=df_pareto[col_name], y=df_pareto['Porcentagem Acumulada'], name="% Acumulada", yaxis='y2', mode='lines+markers', line=dict(color='steelblue')))
        
        fig.update_layout(
            title=title,
            yaxis=dict(title="Horas"),
            yaxis2=dict(title="%", overlaying='y', side='right', range=[0, 105]),
            xaxis=dict(tickangle=-45)
        )
        return fig

    with tab1:
        st.plotly_chart(plot_pareto(corretivas, 'EQUIPAMENTO', 'Pareto de Perdas por Equipamento'), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_pareto(corretivas, 'GRUPO', 'Pareto de Perdas por Grupo'), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_pareto(corretivas, 'SUBGRUPO', 'Pareto de Perdas por Subgrupo'), use_container_width=True)

    # --- 6. ANÁLISE DE CONFIABILIDADE (WEIBULL) E RGA ---
    st.markdown("### 📈 Análise de Confiabilidade e RGA")
    
    if num_falhas > 2:
        # Extraindo tempos entre falhas (TBF) para ajuste de distribuição
        # Simplificação: Usando tempo disponível / falhas por equipamento para simular TBF real
        tbf_data = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
        tbf_data['TBF'] = tbf_data.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
        tbf_clean = tbf_data['TBF'].dropna()
        tbf_clean = tbf_clean[tbf_clean > 0] # Remover valores 0 ou negativos

        if not tbf_clean.empty:
            # Ajuste de Weibull
            shape, loc, scale = weibull_min.fit(tbf_clean, floc=0)
            
            st.write(f"**Parâmetros da Distribuição de Weibull:** Beta (Forma) = {shape:.2f} | Eta (Vida Característica) = {scale:.2f} horas")
            
            # Curva de Confiabilidade R(t)
            t = np.linspace(0.1, tbf_clean.max() * 1.2, 100)
            reliability = weibull_min.sf(t, shape, loc=loc, scale=scale)
            
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name='R(t) - Confiabilidade', line=dict(color='green')))
            fig_rel.update_layout(title="Curva de Confiabilidade - R(t) (Distribuição de Weibull)", xaxis_title="Tempo Operacional (Horas)", yaxis_title="Probabilidade de Sobrevivência")
            
            st.plotly_chart(fig_rel, use_container_width=True)
            
            # Análise RGA - Gráfico de Duane (MTBF Acumulado)
            tbf_data['Tempo Acumulado'] = tbf_data['TOTAL HORAS DECIMAIS'].cumsum() # Aproximação de RGA
            tbf_data['Falhas Acumuladas'] = range(1, len(tbf_data) + 1)
            tbf_data['MTBF Acumulado'] = tbf_data['Tempo Acumulado'] / tbf_data['Falhas Acumuladas']
            
            fig_rga = px.line(tbf_data, x='Tempo Acumulado', y='MTBF Acumulado', title="Reliability Growth Analysis (RGA) - Crescimento da Confiabilidade")
            st.plotly_chart(fig_rga, use_container_width=True)
        else:
            st.warning("Dados de datas insuficientes para calcular os Tempos Entre Falhas (TBF).")
    else:
        st.info("São necessárias pelo menos 3 ocorrências corretivas para gerar a curva de confiabilidade.")

else:
    st.info("Por favor, faça o upload de uma planilha Excel (como a 'PLANILHA CIM.xlsx') na barra lateral para começar a análise.")
