import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st_scipy
import io
import warnings

warnings.filterwarnings('ignore')

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
    
    min_date = df['DATA INÍCIO'].min()
    max_date = df['DATA FIM'].max()
    date_range = st.sidebar.date_input("Período", [min_date, max_date])
    
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

    # Aplicação de filtros
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
    st.markdown("---")
    st.markdown("### 📊 Indicadores Principais (KPIs)")
    
    corretivas = df_filtered[df_filtered['TIPO'] == 'CORRETIVA']
    num_falhas = len(corretivas)
    total_downtime = corretivas['TOTAL HORAS DECIMAIS'].sum()
    
    mttr = total_downtime / num_falhas if num_falhas > 0 else 0
    
    dias_operacao = (date_range[1] - date_range[0]).days if len(date_range) == 2 else 30
    dias_operacao = max(dias_operacao, 1)
    qtd_equipamentos_total = df_filtered['EQUIPAMENTO'].nunique()
    horas_disponiveis_total = dias_operacao * 24 * qtd_equipamentos_total
    mtbf = (horas_disponiveis_total - total_downtime) / num_falhas if num_falhas > 0 else 0
    
    tipos_count = df_filtered['TIPO'].value_counts(normalize=True) * 100
    perc_prev = tipos_count.get('PREVENTIVA', 0)
    perc_corr = tipos_count.get('CORRETIVA', 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MTBF Global (Horas)", f"{mtbf:.2f}")
    col2.metric("MTTR Global (Horas)", f"{mttr:.2f}")
    col3.metric("% Preventiva", f"{perc_prev:.1f}%")
    col4.metric("% Corretiva", f"{perc_corr:.1f}%")

    # --- 5. RESPONSABILIDADE N1 ---
    st.markdown("### 👥 Responsabilidade Nível 1 (%)")
    if 'RESPONSABILIDADE NÍVEL 1' in df_filtered.columns:
        resp_counts = df_filtered['RESPONSABILIDADE NÍVEL 1'].value_counts(normalize=True).reset_index()
        resp_counts.columns = ['Responsabilidade', 'Porcentagem']
        resp_counts['Porcentagem'] = resp_counts['Porcentagem'] * 100
        
        fig_resp = px.bar(resp_counts, x='Porcentagem', y='Responsabilidade', orientation='h', 
                          text=resp_counts['Porcentagem'].apply(lambda x: f'{x:.1f}%'),
                          color='Porcentagem', color_continuous_scale='Blues')
        fig_resp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_resp, use_container_width=True)

    # --- 6. MTBF, MTTR E JACK-KNIFE POR EQUIPAMENTO ---
    st.markdown("---")
    st.markdown("### 🚜 Análise por Equipamento")
    
    if not corretivas.empty:
        equip_stats = corretivas.groupby('EQUIPAMENTO').agg(
            Falhas=('OS', 'count'),
            Downtime=('TOTAL HORAS DECIMAIS', 'sum')
        ).reset_index()
        
        equip_stats['MTTR'] = equip_stats['Downtime'] / equip_stats['Falhas']
        equip_stats['Horas Disponiveis'] = dias_operacao * 24
        equip_stats['MTBF'] = (equip_stats['Horas Disponiveis'] - equip_stats['Downtime']) / equip_stats['Falhas']
        equip_stats['MTBF'] = equip_stats['MTBF'].apply(lambda x: x if x > 0 else 0)

        tab_eq1, tab_eq2, tab_eq3 = st.tabs(["MTBF por Equipamento", "MTTR por Equipamento", "Gráfico de Jack-Knife"])
        
        with tab_eq1:
            fig_mtbf = px.bar(equip_stats.sort_values('MTBF', ascending=False), x='EQUIPAMENTO', y='MTBF', 
                              text=equip_stats['MTBF'].round(1), title="MTBF por Equipamento (Horas)")
            fig_mtbf.update_traces(textposition='outside')
            st.plotly_chart(fig_mtbf, use_container_width=True)
            
        with tab_eq2:
            fig_mttr = px.bar(equip_stats.sort_values('MTTR', ascending=False), x='EQUIPAMENTO', y='MTTR', 
                              text=equip_stats['MTTR'].round(1), title="MTTR por Equipamento (Horas)", color_discrete_sequence=['indianred'])
            fig_mttr.update_traces(textposition='outside')
            st.plotly_chart(fig_mttr, use_container_width=True)

        with tab_eq3:
            mean_falhas = equip_stats['Falhas'].mean()
            mean_mttr = equip_stats['MTTR'].mean()
            
            fig_jk = px.scatter(equip_stats, x='Falhas', y='MTTR', text='EQUIPAMENTO', size='Downtime',
                                title='Jack-Knife: Frequência de Falhas vs MTTR (Tamanho da bolha = Downtime)',
                                labels={'Falhas': 'Número de Falhas', 'MTTR': 'MTTR (Horas)'})
            fig_jk.add_vline(x=mean_falhas, line_dash="dash", line_color="gray", annotation_text="Média de Falhas")
            fig_jk.add_hline(y=mean_mttr, line_dash="dash", line_color="gray", annotation_text="MTTR Médio")
            fig_jk.update_traces(textposition='top center')
            st.plotly_chart(fig_jk, use_container_width=True)

    # --- 7. PERFIL DE PERDAS (PARETO TOP 18) ---
    st.markdown("---")
    st.markdown("### 📉 Perfil de Perdas (Pareto - Top 18)")
    tab_p1, tab_p2, tab_p3 = st.tabs(["Por Equipamento", "Por Grupo", "Por Subgrupo"])
    
    def plot_pareto(data, col_name, title):
        df_pareto = data.groupby(col_name)['TOTAL HORAS DECIMAIS'].sum().reset_index()
        df_pareto = df_pareto.sort_values(by='TOTAL HORAS DECIMAIS', ascending=False).head(18) # Limite ao TOP 18
        
        if df_pareto.empty:
            return go.Figure()

        df_pareto['Porcentagem Acumulada'] = df_pareto['TOTAL HORAS DECIMAIS'].cumsum() / df_pareto['TOTAL HORAS DECIMAIS'].sum() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_pareto[col_name], y=df_pareto['TOTAL HORAS DECIMAIS'], name="Horas Parada", 
                             marker_color='indianred', text=df_pareto['TOTAL HORAS DECIMAIS'].round(1), textposition='auto'))
        fig.add_trace(go.Scatter(x=df_pareto[col_name], y=df_pareto['Porcentagem Acumulada'], name="% Acumulada", 
                                 yaxis='y2', mode='lines+markers', line=dict(color='steelblue')))
        
        fig.update_layout(
            title=title,
            yaxis=dict(title="Horas"),
            yaxis2=dict(title="%", overlaying='y', side='right', range=[0, 105]),
            xaxis=dict(tickangle=-45),
            hovermode="x unified"
        )
        return fig

    with tab_p1:
        st.plotly_chart(plot_pareto(corretivas, 'EQUIPAMENTO', 'Pareto por Equipamento (Top 18)'), use_container_width=True)
    with tab_p2:
        st.plotly_chart(plot_pareto(corretivas, 'GRUPO', 'Pareto por Grupo (Top 18)'), use_container_width=True)
    with tab_p3:
        st.plotly_chart(plot_pareto(corretivas, 'SUBGRUPO', 'Pareto por Subgrupo (Top 18)'), use_container_width=True)

    # --- 8. ANÁLISE DE CONFIABILIDADE (MELHOR DISTRIBUIÇÃO) E RGA ---
    st.markdown("---")
    st.markdown("### 📈 Confiabilidade e Probabilidade de Falha")
    
    if num_falhas > 3:
        tbf_data = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
        tbf_data['TBF'] = tbf_data.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
        tbf_clean = tbf_data['TBF'].dropna()
        tbf_clean = tbf_clean[tbf_clean > 0].values

        if len(tbf_clean) > 3:
            # Seleção da melhor distribuição (SSE - Sum of Squared Errors)
            y_hist, x_hist = np.histogram(tbf_clean, bins='auto', density=True)
            x_mids = (x_hist + np.roll(x_hist, -1))[:-1] / 2.0

            dists = {
                'Weibull': st_scipy.weibull_min,
                'Exponencial': st_scipy.expon,
                'Normal': st_scipy.norm,
                'Lognormal': st_scipy.lognorm
            }
            
            best_sse = np.inf
            best_name = ""
            best_params = None
            
            for name, distribution in dists.items():
                params = distribution.fit(tbf_clean)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf = distribution.pdf(x_mids, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y_hist - pdf, 2.0))
                
                if sse < best_sse:
                    best_sse = sse
                    best_name = name
                    best_params = params

            st.success(f"**Melhor distribuição estatística encontrada:** {best_name} (Aderência com base no menor Erro Quadrático)")

            # Calcular as curvas baseadas na melhor distribuição
            t = np.linspace(0.1, max(tbf_clean) * 1.2, 200)
            dist_obj = dists[best_name]
            arg = best_params[:-2]
            loc = best_params[-2]
            scale = best_params[-1]

            reliability = dist_obj.sf(t, loc=loc, scale=scale, *arg) # R(t)
            prob_failure = dist_obj.cdf(t, loc=loc, scale=scale, *arg) # F(t)
            pdf_vals = dist_obj.pdf(t, loc=loc, scale=scale, *arg) # f(t)
            hazard_rate = pdf_vals / reliability # h(t) = f(t)/R(t)
            hazard_rate[np.isinf(hazard_rate)] = 0 # Evitar divisões por zero

            tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Confiabilidade R(t)", "Probabilidade de Falha F(t)", "Taxa de Falha h(t)", "RGA (Crescimento)"])

            def setup_hover(fig):
                fig.update_layout(hovermode="x unified")
                fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across")
                return fig

            with tab_r1:
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name='Confiabilidade', line=dict(color='green')))
                fig_rel.update_layout(title=f"Curva de Confiabilidade R(t) - {best_name}", xaxis_title="Tempo (Horas)", yaxis_title="Probabilidade")
                st.plotly_chart(setup_hover(fig_rel), use_container_width=True)

            with tab_r2:
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Scatter(x=t, y=prob_failure, mode='lines', name='Prob. Falha', line=dict(color='red')))
                fig_prob.update_layout(title=f"Probabilidade de Falha F(t) - {best_name}", xaxis_title="Tempo (Horas)", yaxis_title="Probabilidade")
                st.plotly_chart(setup_hover(fig_prob), use_container_width=True)

            with tab_r3:
                fig_haz = go.Figure()
                fig_haz.add_trace(go.Scatter(x=t, y=hazard_rate, mode='lines', name='Taxa de Falha', line=dict(color='orange')))
                fig_haz.update_layout(title=f"Taxa de Falha h(t) - {best_name}", xaxis_title="Tempo (Horas)", yaxis_title="Falhas / Hora")
                st.plotly_chart(setup_hover(fig_haz), use_container_width=True)

            with tab_r4:
                # Análise RGA
                tbf_data['Tempo Acumulado'] = tbf_data['TOTAL HORAS DECIMAIS'].cumsum()
                tbf_data['Falhas Acumuladas'] = range(1, len(tbf_data) + 1)
                tbf_data['MTBF Acumulado'] = tbf_data['Tempo Acumulado'] / tbf_data['Falhas Acumuladas']
                
                fig_rga = px.line(tbf_data, x='Tempo Acumulado', y='MTBF Acumulado', title="Reliability Growth Analysis (RGA)", markers=True)
                st.plotly_chart(setup_hover(fig_rga), use_container_width=True)
        else:
            st.warning("Dados de TBF insuficientes (valores positivos).")
    else:
        st.info("São necessárias mais ocorrências corretivas para gerar análises de distribuição.")

else:
    st.info("Por favor, faça o upload de uma planilha Excel na barra lateral.")
