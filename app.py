import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st_scipy
from scipy.special import gamma
import io
import warnings
from streamlit_gsheets import GSheetsConnection

try:
    from lifelines import WeibullFitter, KaplanMeierFitter
    LIFELINES_INSTALLED = True
except ImportError:
    LIFELINES_INSTALLED = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard de Manutenção CIM", layout="wide", page_icon="⚙️")

# --- NAVEGAÇÃO POR ABAS PRINCIPAIS ---
aba_dashboard, aba_plano_acao, aba_lda = st.tabs([
    "📊 Dashboard de OS", 
    "📝 Plano de Ação (5W2H)", 
    "🛠️ Análise LDA (Componentes)"
])

# =====================================================================
# ABA 1: DASHBOARD DE ORDENS DE SERVIÇO
# =====================================================================
with aba_dashboard:
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

    st.title("⚙️ Dashboard de Engenharia de Manutenção")
    uploaded_file = st.sidebar.file_uploader("Carregue sua planilha de ordens de serviço (Ex: PLANILHA CIM)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        df.columns = df.columns.str.replace('\n', ' ').str.strip()
        df['DATA INÍCIO'] = pd.to_datetime(df['DATA INÍCIO'], errors='coerce')
        df['DATA FIM'] = pd.to_datetime(df['DATA FIM'], errors='coerce')
        df['TOTAL HORAS DECIMAIS'] = pd.to_numeric(df['TOTAL HORAS DECIMAIS'], errors='coerce').fillna(0)
        df['TIPO'] = df['TIPO'].astype(str).str.upper().str.strip()

        # --- FILTROS EM CASCATA ---
        st.sidebar.header("Filtros em Cascata")
        st.sidebar.markdown("*Deixe vazio para selecionar todos*")
        
        min_date = df['DATA INÍCIO'].min()
        max_date = df['DATA FIM'].max()
        date_range = st.sidebar.date_input("Período (Afeta apenas KPIs e Pareto)", [min_date, max_date])
        
        df_cascaded = df.copy()
        
        def apply_cascading_filter(df_in, col_name, label):
            if col_name in df_in.columns:
                options = df_in[col_name].dropna().astype(str).unique().tolist()
                options.sort()
                selected = st.sidebar.multiselect(label, options, default=[])
                if len(selected) > 0:
                    df_in = df_in[df_in[col_name].astype(str).isin(selected)]
            return df_in

        df_cascaded = apply_cascading_filter(df_cascaded, 'MODELO EQUIPAMENTO', 'Modelo Equipamento')
        df_cascaded = apply_cascading_filter(df_cascaded, 'EQUIPAMENTO', 'Equipamento')
        df_cascaded = apply_cascading_filter(df_cascaded, 'TIPO', 'Tipo de OS')
        df_cascaded = apply_cascading_filter(df_cascaded, 'RESPONSABILIDADE NÍVEL 1', 'Responsabilidade N1')
        df_cascaded = apply_cascading_filter(df_cascaded, 'RESPONSABILIDADE NÍVEL 2', 'Responsabilidade N2')
        df_cascaded = apply_cascading_filter(df_cascaded, 'GRUPO', 'Grupo')
        df_cascaded = apply_cascading_filter(df_cascaded, 'SUBGRUPO', 'Subgrupo')

        if len(date_range) == 2:
            mask_date = (df_cascaded['DATA INÍCIO'].dt.date >= date_range[0]) & (df_cascaded['DATA INÍCIO'].dt.date <= date_range[1])
            df_filtered = df_cascaded[mask_date]
        else:
            df_filtered = df_cascaded.copy()

        # --- KPIs BÁSICOS ---
        st.markdown("---")
        st.markdown("### 📊 Indicadores Principais (KPIs no Período)")
        
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

        # --- EVOLUÇÃO MENSAL E RESPONSABILIDADE ---
        st.markdown("---")
        col_evol, col_resp = st.columns([2, 1])
        
        with col_evol:
            st.markdown("### 📅 Evolução Mensal Histórica (MTBF e MTTR)")
            corretivas_evol = df_cascaded[df_cascaded['TIPO'] == 'CORRETIVA']
            if not corretivas_evol.empty:
                corretivas_evol['Ano-Mês'] = corretivas_evol['DATA INÍCIO'].dt.strftime('%Y-%m')
                evol_stats = corretivas_evol.groupby('Ano-Mês').agg(Falhas=('OS', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
                evol_stats['MTTR'] = evol_stats['Downtime'] / evol_stats['Falhas']
                
                qtd_equip_evol = df_cascaded['EQUIPAMENTO'].nunique() if not df_cascaded.empty else 1
                evol_stats['MTBF'] = ((730 * qtd_equip_evol) - evol_stats['Downtime']) / evol_stats['Falhas']
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
                fig_resp.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_resp, use_container_width=True)

        # --- MTBF, MTTR E JACK-KNIFE DINÂMICO ---
        st.markdown("---")
        st.markdown("### 🚜 Análise Dinâmica: MTBF, MTTR e Jack-Knife")
        dimensao = st.radio("Selecione a Dimensão de Análise:", ["EQUIPAMENTO", "GRUPO", "SUBGRUPO"], horizontal=True)
        
        if not corretivas.empty and dimensao in corretivas.columns:
            dim_stats = corretivas.groupby(dimensao).agg(Falhas=('OS', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
            equip_count = df_filtered.groupby(dimensao)['EQUIPAMENTO'].nunique().reset_index(name='Qtd_Equip')
            dim_stats = pd.merge(dim_stats, equip_count, on=dimensao)
            
            dim_stats['MTTR'] = dim_stats['Downtime'] / dim_stats['Falhas']
            dim_stats['Horas Disponiveis'] = dias_operacao * 24 * dim_stats['Qtd_Equip']
            dim_stats['MTBF'] = (dim_stats['Horas Disponiveis'] - dim_stats['Downtime']) / dim_stats['Falhas']
            dim_stats['MTBF'] = dim_stats['MTBF'].apply(lambda x: x if x > 0 else 0)

            tab_eq1, tab_eq2, tab_eq3 = st.tabs([f"Jack-Knife por {dimensao.title()}", f"MTBF", f"MTTR"])
            
            with tab_eq1:
                mean_falhas = dim_stats['Falhas'].mean()
                mean_mttr = dim_stats['MTTR'].mean()
                max_f = dim_stats['Falhas'].max() * 1.1 if not dim_stats.empty else 1
                max_m = dim_stats['MTTR'].max() * 1.1 if not dim_stats.empty else 1
                
                # Rótulos ativados e organizados
                fig_jk = px.scatter(dim_stats, x='Falhas', y='MTTR', text=dimensao, size='Downtime',
                                    title=f'Jack-Knife ({dimensao.title()})', opacity=0.8)
                
                fig_jk.update_traces(textposition='top center', textfont=dict(size=11, color='black'), cliponaxis=False)
                
                fig_jk.add_shape(type="rect", x0=0, y0=0, x1=mean_falhas, y1=mean_mttr, fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=mean_falhas, y0=0, x1=max_f, y1=mean_mttr, fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=0, y0=mean_mttr, x1=mean_falhas, y1=max_m, fillcolor="orange", opacity=0.2, layer="below", line_width=0)
                fig_jk.add_shape(type="rect", x0=mean_falhas, y0=mean_mttr, x1=max_f, y1=max_m, fillcolor="red", opacity=0.2, layer="below", line_width=0)

                fig_jk.add_vline(x=mean_falhas, line_dash="dash", line_color="black")
                fig_jk.add_hline(y=mean_mttr, line_dash="dash", line_color="black")
                fig_jk.update_xaxes(range=[0, max_f], title_text="Número de Falhas")
                fig_jk.update_yaxes(range=[0, max_m], title_text="MTTR (Horas)")
                st.plotly_chart(fig_jk, use_container_width=True)

            with tab_eq2:
                fig_mtbf = px.bar(dim_stats.sort_values('MTBF', ascending=False), x=dimensao, y='MTBF', text=dim_stats['MTBF'].round(1), title=f"MTBF por {dimensao.title()}")
                fig_mtbf.update_traces(textposition='outside')
                st.plotly_chart(fig_mtbf, use_container_width=True)
                
            with tab_eq3:
                fig_mttr = px.bar(dim_stats.sort_values('MTTR', ascending=False), x=dimensao, y='MTTR', text=dim_stats['MTTR'].round(1), color_discrete_sequence=['indianred'], title=f"MTTR por {dimensao.title()}")
                fig_mttr.update_traces(textposition='outside')
                st.plotly_chart(fig_mttr, use_container_width=True)

        # --- PARETO (TOP 18) ---
        st.markdown("---")
        st.markdown("### 📉 Perfil de Perdas (Pareto - Top 18)")
        
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

        fig_pareto_equip = plot_pareto(corretivas, 'EQUIPAMENTO', 'Top 18 - Equipamentos')
        tab_p1, tab_p2, tab_p3 = st.tabs(["Por Equipamento", "Por Grupo", "Por Subgrupo"])
        with tab_p1: st.plotly_chart(fig_pareto_equip, use_container_width=True)
        with tab_p2: st.plotly_chart(plot_pareto(corretivas, 'GRUPO', 'Top 18 - Grupos'), use_container_width=True)
        with tab_p3: st.plotly_chart(plot_pareto(corretivas, 'SUBGRUPO', 'Top 18 - Subgrupos'), use_container_width=True)

        # --- CONFIABILIDADE (APENAS WEIBULL) ---
        st.markdown("---")
        st.markdown("### 📈 Confiabilidade e Probabilidade de Falha")
        
        if num_falhas > 3:
            tbf_data = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
            tbf_data['TBF'] = tbf_data.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
            tbf_clean = tbf_data['TBF'].dropna()
            tbf_clean = tbf_clean[tbf_clean > 0].values

            if len(tbf_clean) > 3:
                shape, loc, scale = st_scipy.weibull_min.fit(tbf_clean, floc=0)
                beta = shape
                eta = scale
                
                st.success(f"**Distribuição Utilizada:** Weibull | **Parâmetro de Forma ($\\beta$):** {beta:.3f} | **Vida Característica ($\\eta$):** {eta:.2f} horas")

                t = np.linspace(0.1, max(tbf_clean) * 1.2, 200)
                reliability = st_scipy.weibull_min.sf(t, shape, loc=0, scale=scale)
                prob_failure = st_scipy.weibull_min.cdf(t, shape, loc=0, scale=scale)
                hazard_rate = st_scipy.weibull_min.pdf(t, shape, loc=0, scale=scale) / reliability
                hazard_rate[np.isinf(hazard_rate)] = 0

                tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Confiabilidade R(t)", "Probabilidade de Falha F(t)", "Taxa de Falha h(t)", "RGA"])

                def setup_hover(fig):
                    fig.update_layout(hovermode="x unified")
                    fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across")
                    return fig

                with tab_r1:
                    fig_rel = go.Figure(go.Scatter(x=t, y=reliability, mode='lines', name='Confiabilidade', line=dict(color='green')))
                    fig_rel.update_layout(title="Curva de Confiabilidade R(t) - Weibull", xaxis_title="Tempo (Horas)", yaxis_title="R(t)")
                    st.plotly_chart(setup_hover(fig_rel), use_container_width=True)

                with tab_r2:
                    fig_prob = go.Figure(go.Scatter(x=t, y=prob_failure, mode='lines', name='Prob. Acumulada', line=dict(color='red')))
                    fig_prob.update_layout(title="Probabilidade de Falha F(t) - Weibull", xaxis_title="Tempo (Horas)", yaxis_title="F(t)")
                    st.plotly_chart(setup_hover(fig_prob), use_container_width=True)

                with tab_r3:
                    fig_haz = go.Figure(go.Scatter(x=t, y=hazard_rate, mode='lines', name='Taxa (h(t))', line=dict(color='orange')))
                    fig_haz.update_layout(title="Taxa de Falha h(t) - Weibull", xaxis_title="Tempo (Horas)", yaxis_title="Falhas / Hora")
                    st.plotly_chart(setup_hover(fig_haz), use_container_width=True)

                with tab_r4:
                    tbf_data['Tempo Acumulado'] = tbf_data['TOTAL HORAS DECIMAIS'].cumsum()
                    tbf_data['Falhas Acumuladas'] = range(1, len(tbf_data) + 1)
                    tbf_data['MTBF Acumulado'] = tbf_data['Tempo Acumulado'] / tbf_data['Falhas Acumuladas']
                    fig_rga = px.line(tbf_data, x='Tempo Acumulado', y='MTBF Acumulado', title="RGA - Crescimento da Confiabilidade", markers=True)
                    st.plotly_chart(setup_hover(fig_rga), use_container_width=True)

            else:
                st.warning("Dados de TBF insuficientes (valores positivos).")
    else:
        st.info("Por favor, faça o upload da planilha Excel para iniciar o Dashboard.")

    st.markdown("---")
    st.markdown("💡 **Como gerar o Relatório PDF:** Para evitar lentidão e erros no servidor Cloud, utilize a função nativa do seu navegador. Recolha a barra lateral esquerda e pressione **Ctrl + P** (ou Cmd + P). Selecione 'Salvar como PDF'. Todos os gráficos interativos serão impressos em alta qualidade.")

# =====================================================================
# ABA 2: PLANO DE AÇÃO 5W2H (Google Sheets)
# =====================================================================
with aba_plano_acao:
    st.header("📋 Plano de Ação 5W2H")
    
    url_planilha = "https://docs.google.com/spreadsheets/d/1mrfp_qDdX5_6sJVT5-Gk3NzM3nKqznmlR6rwDsXZN2s/edit?gid=0#gid=0"
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    colunas_5w2h = [
        "NOME DO CONTRATO", "What? (O que será feito)", "Why? (Por que)", 
        "Where? (Onde/Equipamento)", "When? (Prazo)", "Who? (Responsável)", 
        "How? (Como)", "How Much? (Custo Estimado)", "Status"
    ]

    try:
        df_acao = conn.read(spreadsheet=url_planilha)
        if df_acao.empty or len(df_acao.columns) < 2:
            df_acao = pd.DataFrame(columns=colunas_5w2h)
            df_acao.loc[0] = [""] * len(colunas_5w2h)
    except Exception:
        df_acao = pd.DataFrame(columns=colunas_5w2h)
        df_acao.loc[0] = [""] * len(colunas_5w2h)

    st.markdown("Edite a tabela abaixo e clique em **Salvar no Google Sheets**.")
    df_editado = st.data_editor(
        df_acao, 
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Status": st.column_config.SelectboxColumn("Status", options=["Pendente", "Em Andamento", "Concluído", "Atrasado"], required=False)
        }
    )
    
    if st.button("💾 Salvar no Google Sheets"):
        with st.spinner("Salvando..."):
            try:
                conn.update(spreadsheet=url_planilha, data=df_editado)
                st.success("Dados salvos no Google Sheets com sucesso!")
            except Exception:
                st.warning("⚠️ **Autenticação Necessária na Nuvem:** O Streamlit bloqueou a gravação pois a sua Service Account não está configurada nos Secrets. Baixe o plano como backup.")
                
    st.markdown("---")
    st.markdown("💡 **Plano B:** Caso as credenciais do Google Sheets não estejam configuradas na nuvem, faça o download do CSV.")
    csv = df_editado.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Baixar Plano de Ação (CSV Local)", data=csv, file_name='plano_acao_backup.csv', mime='text/csv')

# =====================================================================
# ABA 3: CONTROLE DE COMPONENTES E LDA
# =====================================================================
with aba_lda:
    st.header("🛠️ Análise de Dados de Vida (LDA) - Componentes")
    
    if not LIFELINES_INSTALLED:
        st.error("⚠️ **Atenção:** Biblioteca `lifelines` ausente. Adicione `lifelines` ao `requirements.txt`.")
    else:
        file_lda = st.file_uploader("Carregue a planilha de Controle de Componentes (Ex: SKT110S & SKT130Pro)", type=["xlsx"], key="lda_uploader")
        
        if file_lda is not None:
            df_comp = pd.read_excel(file_lda, sheet_name=0)
            df_comp.columns = df_comp.columns.str.replace('\n', ' ').str.replace('  ', ' ').str.strip()
            
            if 'SITUAÇÃO DO COMPONENTE' in df_comp.columns and 'HORAS TRABALHADAS DO COMPONENTE' in df_comp.columns:
                
                # Filtro de MODELO
                if 'MODELO' in df_comp.columns:
                    modelos_disp = df_comp['MODELO'].dropna().astype(str).unique().tolist()
                    modelo_alvo = st.multiselect("Filtre pelo Modelo:", modelos_disp, default=modelos_disp)
                    if modelo_alvo:
                        df_comp = df_comp[df_comp['MODELO'].astype(str).isin(modelo_alvo)]
                
                df_comp['Status_LDA'] = df_comp['SITUAÇÃO DO COMPONENTE'].apply(lambda x: 1 if isinstance(x, str) and 'falhou' in x.lower() else 0)
                df_comp['Horas_LDA'] = pd.to_numeric(df_comp['HORAS TRABALHADAS DO COMPONENTE'], errors='coerce')
                
                componentes_disp = df_comp['COMPONENTE'].dropna().unique().tolist()
                comp_alvo = st.selectbox("Selecione o Componente para Análise Individual:", componentes_disp)
                
                df_alvo = df_comp[df_comp['COMPONENTE'] == comp_alvo].dropna(subset=['Horas_LDA'])
                df_alvo = df_alvo[df_alvo['Horas_LDA'] > 0]
                
                if len(df_alvo) > 0:
                    falhas_count = df_alvo['Status_LDA'].sum()
                    susp_count = len(df_alvo) - falhas_count
                    
                    st.markdown("### 📋 Tabela Resumo do Componente")
                    cols_to_show = ['MODELO', 'TAG', 'COMPONENTE', 'SITUAÇÃO DO COMPONENTE', 'Horas_LDA']
                    cols_exist = [c for c in cols_to_show if c in df_alvo.columns]
                    st.dataframe(df_alvo[cols_exist].sort_values('Horas_LDA', ascending=False), use_container_width=True)
                    
                    st.write(f"**Amostras no DataSet:** {len(df_alvo)} | **Falhas (Eventos):** {falhas_count} | **Censuras (Em Operação):** {susp_count}")
                    
                    if falhas_count > 0:
                        wf = WeibullFitter()
                        wf.fit(df_alvo['Horas_LDA'], event_observed=df_alvo['Status_LDA'])
                        
                        beta_lda = wf.rho_
                        eta_lda = wf.lambda_
                        mttf_lda = eta_lda * gamma(1 + (1/beta_lda))
                        
                        st.success(f"**Parâmetros Weibull:** $\\beta$ (Forma) = {beta_lda:.2f} | $\\eta$ (Vida Característica) = {eta_lda:.2f}h | **MTTF Estimado:** {mttf_lda:.2f}h")
                        
                        kmf = KaplanMeierFitter()
                        kmf.fit(df_alvo['Horas_LDA'], event_observed=df_alvo['Status_LDA'])
                        
                        t_lda = np.linspace(0.1, df_alvo['Horas_LDA'].max() * 1.2, 100)
                        weibull_surv = wf.survival_function_at_times(t_lda)
                        weibull_cdf = wf.cumulative_density_at_times(t_lda)
                        weibull_haz = wf.hazard_at_times(t_lda)
                        
                        tab_l1, tab_l2, tab_l3 = st.tabs(["Confiabilidade R(t)", "Prob. de Falha F(t)", "Taxa de Falha h(t)"])
                        
                        with tab_l1:
                            fig_l1 = go.Figure()
                            fig_l1.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'], mode='lines', line=dict(shape='hv', color='blue'), name='Kaplan-Meier (Real)'))
                            fig_l1.add_trace(go.Scatter(x=t_lda, y=weibull_surv, mode='lines', line=dict(dash='dash', color='red'), name='Weibull (Ajuste)'))
                            fig_l1.update_layout(title="Aderência da Confiabilidade R(t)", xaxis_title="Horas Operacionais", yaxis_title="R(t)", hovermode="x unified")
                            st.plotly_chart(fig_l1, use_container_width=True)
                            
                        with tab_l2:
                            fig_l2 = go.Figure(go.Scatter(x=t_lda, y=weibull_cdf, mode='lines', line=dict(color='orange')))
                            fig_l2.update_layout(title="Probabilidade Acumulada de Falha F(t)", xaxis_title="Horas Operacionais", yaxis_title="F(t)", hovermode="x unified")
                            st.plotly_chart(fig_l2, use_container_width=True)
                            
                        with tab_l3:
                            fig_l3 = go.Figure(go.Scatter(x=t_lda, y=weibull_haz, mode='lines', line=dict(color='purple')))
                            fig_l3.update_layout(title="Taxa de Falha h(t)", xaxis_title="Horas Operacionais", yaxis_title="h(t)", hovermode="x unified")
                            st.plotly_chart(fig_l3, use_container_width=True)
                    else:
                        st.warning("Não há falhas registradas. Não é possível calcular os parâmetros estatísticos de vida.")
                
                # --- MAPA DE CALOR COM FILTROS E TABELA DE VIDA ÚTIL ---
                st.markdown("---")
                st.markdown("### 🔥 Mapa de Calor: Vida Útil Restante da Frota")
                
                component_mttf = {}
                for comp in df_comp['COMPONENTE'].dropna().unique():
                    df_c = df_comp[(df_comp['COMPONENTE'] == comp) & (df_comp['Horas_LDA'] > 0)]
                    if df_c['Status_LDA'].sum() > 0:
                        try:
                            wf_heat = WeibullFitter()
                            wf_heat.fit(df_c['Horas_LDA'], event_observed=df_c['Status_LDA'])
                            mttf_heat = wf_heat.lambda_ * gamma(1 + (1/wf_heat.rho_))
                            component_mttf[comp] = mttf_heat
                        except Exception:
                            pass

                df_ativos = df_comp[(df_comp['Status_LDA'] == 0) & (df_comp['Horas_LDA'] > 0)].copy()
                df_ativos['MTTF'] = df_ativos['COMPONENTE'].map(component_mttf)
                df_ativos = df_ativos.dropna(subset=['MTTF'])

                if not df_ativos.empty and 'TAG' in df_ativos.columns:
                    df_ativos['Vida_Consumida_%'] = (df_ativos['Horas_LDA'] / df_ativos['MTTF']) * 100
                    df_ativos['EQUIP'] = df_ativos['TAG'].astype(str).apply(lambda x: x.split(' ')[0])
                    
                    st.markdown("#### Filtros do Mapa de Calor")
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        max_vida = float(df_ativos['Vida_Consumida_%'].max()) if not df_ativos.empty else 100.0
                        faixa_vida = st.slider(
                            "Selecione a faixa de % de Vida Consumida:", 
                            min_value=0.0, 
                            max_value=max(200.0, max_vida), 
                            value=(0.0, max(100.0, max_vida))
                        )
                    
                    with col_f2:
                        comps_heatmap = df_ativos['COMPONENTE'].unique().tolist()
                        selecao_comps = st.multiselect("Selecione os Componentes para exibir:", comps_heatmap, default=comps_heatmap)

                    df_heat_filtered = df_ativos[
                        (df_ativos['Vida_Consumida_%'] >= faixa_vida[0]) & 
                        (df_ativos['Vida_Consumida_%'] <= faixa_vida[1]) &
                        (df_ativos['COMPONENTE'].isin(selecao_comps))
                    ]

                    if not df_heat_filtered.empty:
                        heatmap_data = df_heat_filtered.pivot_table(index='EQUIP', columns='COMPONENTE', values='Vida_Consumida_%', aggfunc='mean').fillna(0)
                        
                        fig_heat = px.imshow(
                            heatmap_data, 
                            text_auto=".1f", 
                            aspect="auto", 
                            color_continuous_scale="RdYlGn_r",
                            title="Mapa de Calor: Porcentagem (%) do MTTF Consumida",
                        )
                        fig_heat.update_layout(coloraxis_colorbar=dict(title="% Consumida", thicknessmode="pixels", thickness=15))
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                        # Tabela de Vida Útil (MTTF) abaixo do mapa
                        st.markdown("#### 🕒 Tabela de Vida Útil (MTTF Estimado)")
                        df_mttf_display = pd.DataFrame(list(component_mttf.items()), columns=['Componente', 'MTTF (Horas)'])
                        df_mttf_display = df_mttf_display[df_mttf_display['Componente'].isin(selecao_comps)].sort_values('MTTF (Horas)')
                        df_mttf_display['MTTF (Horas)'] = df_mttf_display['MTTF (Horas)'].round(2)
                        st.dataframe(df_mttf_display, use_container_width=True)
                    else:
                        st.warning("Nenhum dado encontrado para os filtros selecionados.")
                else:
                    st.info("É necessário haver falhas computadas nos componentes para determinar o MTTF e compará-los com as peças ativas no mapa de calor.")

            else:
                st.error("A planilha não possui as colunas 'SITUAÇÃO DO COMPONENTE' e/ou 'HORAS TRABALHADAS DO COMPONENTE'.")
