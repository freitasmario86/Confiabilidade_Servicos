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

# --- NAVEGAÇÃO POR ABAS PRINCIPAIS ---
aba_dashboard, aba_plano_acao = st.tabs(["📊 Dashboard de Confiabilidade", "📝 Plano de Ação (5W2H)"])

with aba_dashboard:
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
    st.title("⚙️ Dashboard de Engenharia de Manutenção")
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

        # --- 3. FILTROS LATERAIS (EM CASCATA) ---
        st.sidebar.header("Filtros em Cascata")
        st.sidebar.markdown("*Deixe vazio para selecionar todos*")
        
        min_date = df['DATA INÍCIO'].min()
        max_date = df['DATA FIM'].max()
        date_range = st.sidebar.date_input("Período", [min_date, max_date])
        
        if len(date_range) == 2:
            df_filtered = df[(df['DATA INÍCIO'].dt.date >= date_range[0]) & (df['DATA INÍCIO'].dt.date <= date_range[1])]
        else:
            df_filtered = df.copy()

        def apply_cascading_filter(df_in, col_name, label):
            if col_name in df_in.columns:
                options = df_in[col_name].dropna().astype(str).unique().tolist()
                options.sort()
                selected = st.sidebar.multiselect(label, options, default=[])
                if len(selected) > 0:
                    df_in = df_in[df_in[col_name].astype(str).isin(selected)]
            return df_in

        df_filtered = apply_cascading_filter(df_filtered, 'MODELO EQUIPAMENTO', 'Modelo Equipamento')
        df_filtered = apply_cascading_filter(df_filtered, 'EQUIPAMENTO', 'Equipamento')
        df_filtered = apply_cascading_filter(df_filtered, 'TIPO', 'Tipo de OS')
        df_filtered = apply_cascading_filter(df_filtered, 'RESPONSABILIDADE NÍVEL 1', 'Responsabilidade N1')
        df_filtered = apply_cascading_filter(df_filtered, 'RESPONSABILIDADE NÍVEL 2', 'Responsabilidade N2')
        df_filtered = apply_cascading_filter(df_filtered, 'GRUPO', 'Grupo')
        df_filtered = apply_cascading_filter(df_filtered, 'SUBGRUPO', 'Subgrupo')

        # --- 4. KPIs BÁSICOS ---
        st.markdown("---")
        st.markdown("### 📊 Indicadores Principais (KPIs)")
        
        corretivas = df_filtered[df_filtered['TIPO'] == 'CORRETIVA']
        num_falhas = len(corretivas)
        total_downtime = corretivas['TOTAL HORAS DECIMAIS'].sum()
        
        mttr = total_downtime / num_falhas if num_falhas > 0 else 0
        
        dias_operacao = (date_range[1] - date_range[0]).days if len(date_range) == 2 else 30
        dias_operacao = max(dias_operacao, 1)
        qtd_equipamentos_total = df_filtered['EQUIPAMENTO'].nunique() if not df_filtered.empty else 1
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

        # --- 5. EVOLUÇÃO MENSAL (MTBF / MTTR) ---
        st.markdown("---")
        st.markdown("### 📅 Evolução Temporal (MTBF e MTTR)")
        if not corretivas.empty:
            # Criar coluna Ano-Mês
            corretivas['Ano-Mês'] = corretivas['DATA INÍCIO'].dt.strftime('%Y-%m')
            
            evol_stats = corretivas.groupby('Ano-Mês').agg(
                Falhas=('OS', 'count'),
                Downtime=('TOTAL HORAS DECIMAIS', 'sum')
            ).reset_index()
            
            evol_stats['MTTR'] = evol_stats['Downtime'] / evol_stats['Falhas']
            # Assumindo uma média de 730 horas por mês por equipamento para o cálculo simplificado mensal
            evol_stats['MTBF'] = ((730 * qtd_equipamentos_total) - evol_stats['Downtime']) / evol_stats['Falhas']
            evol_stats['MTBF'] = evol_stats['MTBF'].apply(lambda x: x if x > 0 else 0)
            
            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(x=evol_stats['Ano-Mês'], y=evol_stats['MTBF'], mode='lines+markers', name='MTBF (Horas)', line=dict(color='green')))
            fig_evol.add_trace(go.Scatter(x=evol_stats['Ano-Mês'], y=evol_stats['MTTR'], mode='lines+markers', name='MTTR (Horas)', yaxis='y2', line=dict(color='red')))
            
            fig_evol.update_layout(
                title="Tendência de MTBF e MTTR ao longo do tempo (Escalável pelos filtros)",
                yaxis=dict(title="MTBF (Horas)", side='left'),
                yaxis2=dict(title="MTTR (Horas)", overlaying='y', side='right'),
                hovermode="x unified",
                xaxis=dict(showspikes=True)
            )
            st.plotly_chart(fig_evol, use_container_width=True)

        # --- 6. MTBF, MTTR E JACK-KNIFE POR EQUIPAMENTO ---
        st.markdown("---")
        st.markdown("### 🚜 Análise por Equipamento e Gráfico de Jack-Knife")
        
        if not corretivas.empty:
            equip_stats = corretivas.groupby('EQUIPAMENTO').agg(
                Falhas=('OS', 'count'),
                Downtime=('TOTAL HORAS DECIMAIS', 'sum')
            ).reset_index()
            
            equip_stats['MTTR'] = equip_stats['Downtime'] / equip_stats['Falhas']
            equip_stats['MTBF'] = ((dias_operacao * 24) - equip_stats['Downtime']) / equip_stats['Falhas']
            equip_stats['MTBF'] = equip_stats['MTBF'].apply(lambda x: x if x > 0 else 0)

            tab_eq1, tab_eq2, tab_eq3 = st.tabs(["Gráfico de Jack-Knife", "MTBF por Equipamento", "MTTR por Equipamento"])
            
            with tab_eq1:
                mean_falhas = equip_stats['Falhas'].mean()
                mean_mttr = equip_stats['MTTR'].mean()
                max_f = equip_stats['Falhas'].max() * 1.1 if not equip_stats.empty else 1
                max_m = equip_stats['MTTR'].max() * 1.1 if not equip_stats.empty else 1
                
                fig_jk = px.scatter(equip_stats, x='Falhas', y='MTTR', text='EQUIPAMENTO', size='Downtime',
                                    title='Jack-Knife: Frequência de Falhas vs MTTR (Bolha = Horas Paradas)',
                                    labels={'Falhas': 'Número de Falhas', 'MTTR': 'MTTR (Horas)'})
                
                # Quadrantes Coloridos
                fig_jk.add_shape(type="rect", x0=0, y0=0, x1=mean_falhas, y1=mean_mttr, fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=mean_falhas, y0=0, x1=max_f, y1=mean_mttr, fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=0, y0=mean_mttr, x1=mean_falhas, y1=max_m, fillcolor="orange", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=mean_falhas, y0=mean_mttr, x1=max_f, y1=max_m, fillcolor="red", opacity=0.2, layer="below", line_width=0)

                fig_jk.add_vline(x=mean_falhas, line_dash="dash", line_color="black")
                fig_jk.add_hline(y=mean_mttr, line_dash="dash", line_color="black")
                
                fig_jk.update_traces(textposition='top center')
                fig_jk.update_xaxes(range=[0, max_f])
                fig_jk.update_yaxes(range=[0, max_m])
                st.plotly_chart(fig_jk, use_container_width=True)
                
                st.markdown("""
                **Interpretação dos Quadrantes (Jack-Knife):**
                * 🔴 **Superior Direito (Vermelho - Crítico):** Alta frequência e alto tempo de reparo. **Ação:** Projetos de redesenho, substituição ou overhaul.
                * 🟠 **Superior Esquerdo (Laranja - Crônico):** Baixa frequência, mas reparo demorado. **Ação:** Treinamento de equipe, aquisição de ferramentas especiais ou revisão de estoque de peças críticas.
                * 🟡 **Inferior Direito (Amarelo - Repetitivo):** Quebra muito, mas conserto é rápido. **Ação:** Investigar causa raiz para evitar micro-paradas recorrentes (melhorar a rotina de inspeção).
                * 🟢 **Inferior Esquerdo (Verde - Normal):** Baixa frequência e reparo rápido. **Ação:** Manter os planos de manutenção atuais.
                """)

            with tab_eq2:
                fig_mtbf = px.bar(equip_stats.sort_values('MTBF', ascending=False), x='EQUIPAMENTO', y='MTBF', 
                                  text=equip_stats['MTBF'].round(1), title="MTBF por Equipamento (Horas)")
                fig_mtbf.update_traces(textposition='outside')
                st.plotly_chart(fig_mtbf, use_container_width=True)
                
            with tab_eq3:
                fig_mttr = px.bar(equip_stats.sort_values('MTTR', ascending=False), x='EQUIPAMENTO', y='MTTR', 
                                  text=equip_stats['MTTR'].round(1), title="MTTR por Equipamento (Horas)", color_discrete_sequence=['indianred'])
                fig_mttr.update_traces(textposition='outside')
                st.plotly_chart(fig_mttr, use_container_width=True)

        # O restante do código (Pareto e Distribuição de Probabilidades) continua o mesmo da versão anterior...
        # (Para manter a resposta concisa, não repeti o bloco de Pareto e Weibull que já está funcionando bem, 
        # você deve mantê-los logo abaixo deste trecho no seu arquivo).

    else:
        st.info("Por favor, faça o upload da planilha Excel para iniciar o Dashboard.")

# --- ABA DO PLANO DE AÇÃO 5W2H ---
with aba_plano_acao:
    st.header("📋 Acompanhamento de Ações - 5W2H")
    st.markdown("""
    Nesta tela você pode registrar e acompanhar as ações definidas a partir das análises do Dashboard.
    *Nota: Em ambiente Cloud, conecte este dataframe ao **Google Sheets** ou a um banco de dados para salvar as alterações definitivamente.*
    """)
    
    # Inicializa o dataframe vazio na memória de sessão (se não existir)
    if 'plano_acao' not in st.session_state:
        df_acao = pd.DataFrame(columns=[
            "What? (O que será feito)", 
            "Why? (Por que)", 
            "Where? (Onde/Equipamento)", 
            "When? (Prazo)", 
            "Who? (Responsável)", 
            "How? (Como)", 
            "How Much? (Custo Estimado)", 
            "Status"
        ])
        # Adiciona uma linha de exemplo
        df_acao.loc[0] = ["Revisão do motor", "Alto tempo de inatividade (Jack-Knife Vermelho)", "CS32", "15/10/2026", "João Manutenção", "Realizar overhaul", "R$ 15.000", "Pendente"]
        st.session_state.plano_acao = df_acao

    # Usando o data_editor nativo do Streamlit para permitir edições
    st.session_state.plano_acao = st.data_editor(
        st.session_state.plano_acao, 
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                help="Selecione o status da ação",
                options=["Pendente", "Em Andamento", "Concluído", "Atrasado"],
                required=True,
            )
        }
    )
    
    if st.button("Salvar Plano de Ação (Exportar para CSV)"):
        csv = st.session_state.plano_acao.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar Arquivo CSV",
            data=csv,
            file_name='plano_acao_5w2h.csv',
            mime='text/csv',
        )
