import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st_scipy
import io
import warnings
from streamlit_gsheets import GSheetsConnection

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
        
        df.columns = df.columns.str.replace('\n', ' ').str.strip()
        df['DATA INÍCIO'] = pd.to_datetime(df['DATA INÍCIO'], errors='coerce')
        df['DATA FIM'] = pd.to_datetime(df['DATA FIM'], errors='coerce')
        df['TOTAL HORAS DECIMAIS'] = pd.to_numeric(df['TOTAL HORAS DECIMAIS'], errors='coerce').fillna(0)
        df['TIPO'] = df['TIPO'].astype(str).str.upper().str.strip()

        # --- 3. FILTROS LATERAIS EM CASCATA ---
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

        # --- 5. EVOLUÇÃO MENSAL E RESPONSABILIDADE ---
        st.markdown("---")
        col_evol, col_resp = st.columns([2, 1])
        
        with col_evol:
            st.markdown("### 📅 Evolução Mensal (MTBF e MTTR)")
            if not corretivas.empty:
                corretivas['Ano-Mês'] = corretivas['DATA INÍCIO'].dt.strftime('%Y-%m')
                evol_stats = corretivas.groupby('Ano-Mês').agg(Falhas=('OS', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
                evol_stats['MTTR'] = evol_stats['Downtime'] / evol_stats['Falhas']
                evol_stats['MTBF'] = ((730 * qtd_equipamentos_total) - evol_stats['Downtime']) / evol_stats['Falhas']
                evol_stats['MTBF'] = evol_stats['MTBF'].apply(lambda x: x if x > 0 else 0)
                
                fig_evol = go.Figure()
                fig_evol.add_trace(go.Scatter(x=evol_stats['Ano-Mês'], y=evol_stats['MTBF'], mode='lines+markers', name='MTBF (Horas)', line=dict(color='green')))
                fig_evol.add_trace(go.Scatter(x=evol_stats['Ano-Mês'], y=evol_stats['MTTR'], mode='lines+markers', name='MTTR (Horas)', yaxis='y2', line=dict(color='red')))
                fig_evol.update_layout(yaxis=dict(title="MTBF"), yaxis2=dict(title="MTTR", overlaying='y', side='right'), hovermode="x unified")
                st.plotly_chart(fig_evol, use_container_width=True)

        with col_resp:
            st.markdown("### 👥 Resp. N1 (%)")
            if 'RESPONSABILIDADE NÍVEL 1' in df_filtered.columns and not df_filtered.empty:
                resp_counts = df_filtered['RESPONSABILIDADE NÍVEL 1'].value_counts(normalize=True).reset_index()
                resp_counts.columns = ['Resp', 'Porcentagem']
                resp_counts['Porcentagem'] = resp_counts['Porcentagem'] * 100
                fig_resp = px.bar(resp_counts, x='Porcentagem', y='Resp', orientation='h', text=resp_counts['Porcentagem'].apply(lambda x: f'{x:.1f}%'), color='Porcentagem', color_continuous_scale='Blues')
                st.plotly_chart(fig_resp, use_container_width=True)

        # --- 6. MTBF, MTTR E JACK-KNIFE POR EQUIPAMENTO ---
        st.markdown("---")
        st.markdown("### 🚜 Análise por Equipamento e Jack-Knife")
        
        if not corretivas.empty:
            equip_stats = corretivas.groupby('EQUIPAMENTO').agg(Falhas=('OS', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
            equip_stats['MTTR'] = equip_stats['Downtime'] / equip_stats['Falhas']
            equip_stats['MTBF'] = ((dias_operacao * 24) - equip_stats['Downtime']) / equip_stats['Falhas']
            equip_stats['MTBF'] = equip_stats['MTBF'].apply(lambda x: x if x > 0 else 0)

            tab_eq1, tab_eq2, tab_eq3 = st.tabs(["Jack-Knife", "MTBF", "MTTR"])
            
            with tab_eq1:
                mean_falhas = equip_stats['Falhas'].mean()
                mean_mttr = equip_stats['MTTR'].mean()
                max_f = equip_stats['Falhas'].max() * 1.1 if not equip_stats.empty else 1
                max_m = equip_stats['MTTR'].max() * 1.1 if not equip_stats.empty else 1
                
                fig_jk = px.scatter(equip_stats, x='Falhas', y='MTTR', text='EQUIPAMENTO', size='Downtime',
                                    title='Jack-Knife (Tamanho da bolha = Horas Paradas)', labels={'Falhas': 'Número de Falhas', 'MTTR': 'MTTR (Horas)'})
                
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
                * 🔴 **Vermelho (Crítico):** Alta frequência + Alto tempo de reparo. Ação: Redesenho/Substituição.
                * 🟠 **Laranja (Crônico):** Baixa frequência + Alto MTTR. Ação: Treinamento, ferramentas, estoque de peças.
                * 🟡 **Amarelo (Repetitivo):** Quebra muito + Conserto rápido. Ação: Investigar causa raiz (micro-paradas).
                * 🟢 **Verde (Normal):** Sob controle operacional.
                """)

            with tab_eq2:
                fig_mtbf = px.bar(equip_stats.sort_values('MTBF', ascending=False), x='EQUIPAMENTO', y='MTBF', text=equip_stats['MTBF'].round(1), title="MTBF por Equipamento")
                fig_mtbf.update_traces(textposition='outside')
                st.plotly_chart(fig_mtbf, use_container_width=True)
                
            with tab_eq3:
                fig_mttr = px.bar(equip_stats.sort_values('MTTR', ascending=False), x='EQUIPAMENTO', y='MTTR', text=equip_stats['MTTR'].round(1), color_discrete_sequence=['indianred'], title="MTTR por Equipamento")
                fig_mttr.update_traces(textposition='outside')
                st.plotly_chart(fig_mttr, use_container_width=True)

        # --- 7. PERFIL DE PERDAS (PARETO TOP 18) ---
        st.markdown("---")
        st.markdown("### 📉 Perfil de Perdas (Pareto - Top 18)")
        tab_p1, tab_p2, tab_p3 = st.tabs(["Por Equipamento", "Por Grupo", "Por Subgrupo"])
        
        def plot_pareto(data, col_name, title):
            df_pareto = data.groupby(col_name)['TOTAL HORAS DECIMAIS'].sum().reset_index()
            df_pareto = df_pareto.sort_values(by='TOTAL HORAS DECIMAIS', ascending=False).head(18)
            if df_pareto.empty: return go.Figure()

            df_pareto['Porcentagem Acumulada'] = df_pareto['TOTAL HORAS DECIMAIS'].cumsum() / df_pareto['TOTAL HORAS DECIMAIS'].sum() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_pareto[col_name], y=df_pareto['TOTAL HORAS DECIMAIS'], name="Horas Parada", marker_color='indianred', text=df_pareto['TOTAL HORAS DECIMAIS'].round(1), textposition='auto'))
            fig.add_trace(go.Scatter(x=df_pareto[col_name], y=df_pareto['Porcentagem Acumulada'], name="% Acumulada", yaxis='y2', mode='lines+markers', line=dict(color='steelblue')))
            fig.update_layout(title=title, yaxis=dict(title="Horas"), yaxis2=dict(title="%", overlaying='y', side='right', range=[0, 105]), hovermode="x unified")
            return fig

        with tab_p1: st.plotly_chart(plot_pareto(corretivas, 'EQUIPAMENTO', 'Top 18 - Equipamentos'), use_container_width=True)
        with tab_p2: st.plotly_chart(plot_pareto(corretivas, 'GRUPO', 'Top 18 - Grupos'), use_container_width=True)
        with tab_p3: st.plotly_chart(plot_pareto(corretivas, 'SUBGRUPO', 'Top 18 - Subgrupos'), use_container_width=True)

        # --- 8. ANÁLISE DE CONFIABILIDADE ---
        st.markdown("---")
        st.markdown("### 📈 Confiabilidade e Probabilidade de Falha")
        
        if num_falhas > 3:
            tbf_data = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
            tbf_data['TBF'] = tbf_data.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
            tbf_clean = tbf_data['TBF'].dropna()
            tbf_clean = tbf_clean[tbf_clean > 0].values

            if len(tbf_clean) > 3:
                y_hist, x_hist = np.histogram(tbf_clean, bins='auto', density=True)
                x_mids = (x_hist + np.roll(x_hist, -1))[:-1] / 2.0

                dists = {'Weibull': st_scipy.weibull_min, 'Exponencial': st_scipy.expon, 'Normal': st_scipy.norm, 'Lognormal': st_scipy.lognorm}
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
                        best_sse = sse; best_name = name; best_params = params

                st.success(f"**Melhor distribuição estatística encontrada:** {best_name}")

                t = np.linspace(0.1, max(tbf_clean) * 1.2, 200)
                dist_obj = dists[best_name]
                arg = best_params[:-2]; loc = best_params[-2]; scale = best_params[-1]

                reliability = dist_obj.sf(t, loc=loc, scale=scale, *arg)
                prob_failure = dist_obj.cdf(t, loc=loc, scale=scale, *arg)
                hazard_rate = dist_obj.pdf(t, loc=loc, scale=scale, *arg) / reliability
                hazard_rate[np.isinf(hazard_rate)] = 0

                tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Confiabilidade R(t)", "Probabilidade de Falha F(t)", "Taxa de Falha h(t)", "RGA"])

                def setup_hover(fig):
                    fig.update_layout(hovermode="x unified")
                    fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across")
                    return fig

                with tab_r1:
                    fig_rel = go.Figure(go.Scatter(x=t, y=reliability, mode='lines', name='Confiabilidade', line=dict(color='green')))
                    fig_rel.update_layout(title="Curva de Confiabilidade R(t)")
                    st.plotly_chart(setup_hover(fig_rel), use_container_width=True)

                with tab_r2:
                    fig_prob = go.Figure(go.Scatter(x=t, y=prob_failure, mode='lines', name='Prob. Acumulada', line=dict(color='red')))
                    fig_prob.update_layout(title="Probabilidade de Falha F(t)")
                    st.plotly_chart(setup_hover(fig_prob), use_container_width=True)

                with tab_r3:
                    fig_haz = go.Figure(go.Scatter(x=t, y=hazard_rate, mode='lines', name='Taxa (h(t))', line=dict(color='orange')))
                    fig_haz.update_layout(title="Taxa de Falha h(t)")
                    st.plotly_chart(setup_hover(fig_haz), use_container_width=True)

                with tab_r4:
                    tbf_data['Tempo Acumulado'] = tbf_data['TOTAL HORAS DECIMAIS'].cumsum()
                    tbf_data['Falhas Acumuladas'] = range(1, len(tbf_data) + 1)
                    tbf_data['MTBF Acumulado'] = tbf_data['Tempo Acumulado'] / tbf_data['Falhas Acumuladas']
                    fig_rga = px.line(tbf_data, x='Tempo Acumulado', y='MTBF Acumulado', title="RGA - Crescimento da Confiabilidade", markers=True)
                    st.plotly_chart(setup_hover(fig_rga), use_container_width=True)

        else:
            st.info("São necessárias mais ocorrências para análises de distribuição e confiabilidade.")

    else:
        st.info("Por favor, faça o upload da planilha Excel para iniciar o Dashboard.")

# --- ABA DO PLANO DE AÇÃO 5W2H (Sincronizado via Google Sheets) ---
with aba_plano_acao:
    st.header("📋 Plano de Ação 5W2H (Google Sheets)")
    
    url_planilha = "https://docs.google.com/spreadsheets/d/1mrfp_qDdX5_6sJVT5-Gk3NzM3nKqznmlR6rwDsXZN2s/edit?gid=0#gid=0"
    
    # Estabelece conexão nativa com o Google Sheets
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    colunas_5w2h = [
        "NOME DO CONTRATO", "What? (O que será feito)", "Why? (Por que)", 
        "Where? (Onde/Equipamento)", "When? (Prazo)", "Who? (Responsável)", 
        "How? (Como)", "How Much? (Custo Estimado)", "Status"
    ]

    try:
        # Tenta ler os dados da planilha
        df_acao = conn.read(spreadsheet=url_planilha)
        # Se a planilha estiver completamente vazia, inicia com as colunas certas
        if df_acao.empty or "What? (O que será feito)" not in df_acao.columns:
            df_acao = pd.DataFrame(columns=colunas_5w2h)
    except Exception as e:
        st.warning("Não foi possível ler a planilha ou ela está vazia. Certifique-se de que a configuração 'secrets.toml' está correta.")
        df_acao = pd.DataFrame(columns=colunas_5w2h)

    # Editor de dados interativo
    st.markdown("Edite a tabela abaixo e clique no botão **Salvar no Google Sheets** para sincronizar suas ações em tempo real.")
    df_editado = st.data_editor(
        df_acao, 
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Pendente", "Em Andamento", "Concluído", "Atrasado"],
                required=True,
            )
        }
    )
    
    if st.button("💾 Salvar no Google Sheets"):
        with st.spinner("Salvando..."):
            # O st-gsheets-connection sobrescreve os dados atualizados de volta na planilha
            conn.update(spreadsheet=url_planilha, data=df_editado)
            st.success("Dados salvos no Google Sheets com sucesso!")
