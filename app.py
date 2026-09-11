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

# Importações de IA e Confiabilidade Avançada
try:
    from lifelines import WeibullFitter, KaplanMeierFitter
    LIFELINES_INSTALLED = True
except ImportError:
    LIFELINES_INSTALLED = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_INSTALLED = True
except ImportError:
    SKLEARN_INSTALLED = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard de Manutenção CIM", layout="wide", page_icon="⚙️")

# =====================================================================
# FUNÇÕES DE CACHE (Para resolver a lentidão e quedas)
# =====================================================================
@st.cache_data
def carregar_dados_os(file):
    df = pd.read_excel(file, sheet_name=0)
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    df['DATA INÍCIO'] = pd.to_datetime(df['DATA INÍCIO'], errors='coerce')
    df['DATA FIM'] = pd.to_datetime(df['DATA FIM'], errors='coerce')
    df['TOTAL HORAS DECIMAIS'] = pd.to_numeric(df['TOTAL HORAS DECIMAIS'], errors='coerce').fillna(0)
    df['TIPO'] = df['TIPO'].astype(str).str.upper().str.strip()
    return df

@st.cache_data
def calcular_mttf_componentes(df_comp):
    component_mttf = {}
    if not LIFELINES_INSTALLED:
        return component_mttf
    for comp in df_comp['COMPONENTE'].dropna().unique():
        df_c = df_comp[(df_comp['COMPONENTE'] == comp) & (df_comp['Horas_LDA'] > 0)]
        if df_c['Status_LDA'].sum() > 0:
            try:
                wf_heat = WeibullFitter()
                wf_heat.fit(df_c['Horas_LDA'], event_observed=df_c['Status_LDA'])
                mttf = wf_heat.lambda_ * gamma(1 + (1/wf_heat.rho_))
                component_mttf[comp] = mttf
            except Exception:
                pass
    return component_mttf

@st.cache_resource
def treinar_modelo_ml(ml_data):
    le_modelo = LabelEncoder()
    le_equip = LabelEncoder()
    le_grupo = LabelEncoder()
    le_resp = LabelEncoder()
    
    ml_data['Modelo_Enc'] = le_modelo.fit_transform(ml_data['MODELO EQUIPAMENTO'])
    ml_data['Equip_Enc'] = le_equip.fit_transform(ml_data['EQUIPAMENTO'])
    ml_data['Grupo_Enc'] = le_grupo.fit_transform(ml_data['GRUPO'])
    ml_data['Resp_Enc'] = le_resp.fit_transform(ml_data['RESPONSABILIDADE NÍVEL 1'])
    
    X = ml_data[['Modelo_Enc', 'Equip_Enc', 'Grupo_Enc', 'Resp_Enc']]
    y = ml_data['TOTAL HORAS DECIMAIS']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model, le_modelo, le_equip, le_grupo, le_resp

# --- NAVEGAÇÃO POR ABAS PRINCIPAIS ---
aba_dashboard, aba_ia, aba_plano_acao, aba_lda = st.tabs([
    "📊 Dashboard de OS", 
    "🧠 IA & Confiabilidade Avançada",
    "📝 Plano de Ação (5W2H)", 
    "🛠️ Análise LDA (Componentes)"
])

# =====================================================================
# ABA 1: DASHBOARD DE ORDENS DE SERVIÇO
# =====================================================================
with aba_dashboard:
    st.title("⚙️ Dashboard de Engenharia de Manutenção")
    uploaded_file = st.sidebar.file_uploader("Carregue a Planilha de Ordens de Serviço (OS)", type=["xlsx"], key="os_file")

    if uploaded_file is not None:
        df = carregar_dados_os(uploaded_file)

        # --- FILTROS EM CASCATA ---
        st.sidebar.header("Filtros em Cascata")
        st.sidebar.markdown("*Deixe vazio para selecionar todos*")
        
        min_date = df['DATA INÍCIO'].min()
        max_date = df['DATA FIM'].max()
        date_range = st.sidebar.date_input("Período (Afeta apenas KPIs Operacionais)", [min_date, max_date])
        
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
        df_cascaded = apply_cascading_filter(df_cascaded, 'GRUPO', 'Grupo')
        df_cascaded = apply_cascading_filter(df_cascaded, 'SUBGRUPO', 'Subgrupo')

        if len(date_range) == 2:
            mask_date = (df_cascaded['DATA INÍCIO'].dt.date >= date_range[0]) & (df_cascaded['DATA INÍCIO'].dt.date <= date_range[1])
            df_filtered = df_cascaded[mask_date]
        else:
            df_filtered = df_cascaded.copy()

        # --- KPIs BÁSICOS E % INDEPENDENTE ---
        st.markdown("---")
        st.markdown("### 📊 Indicadores Principais")
        
        corretivas = df_filtered[df_filtered['TIPO'] == 'CORRETIVA']
        num_falhas = len(corretivas)
        total_downtime = corretivas['TOTAL HORAS DECIMAIS'].sum()
        mttr = total_downtime / num_falhas if num_falhas > 0 else 0
        
        dias_operacao = (date_range[1] - date_range[0]).days if len(date_range) == 2 else 30
        dias_operacao = max(dias_operacao, 1)
        qtd_equipamentos_total = df_filtered['EQUIPAMENTO'].nunique() if not df_filtered.empty else 1
        horas_disponiveis_total = dias_operacao * 24 * qtd_equipamentos_total
        mtbf = (horas_disponiveis_total - total_downtime) / num_falhas if num_falhas > 0 else 0
        
        # % Corretiva e Preventiva (LIDO DA BASE BRUTA, Independente de Filtros)
        tipos_count_global = df['TIPO'].value_counts(normalize=True) * 100
        perc_prev = tipos_count_global.get('PREVENTIVA', 0)
        perc_corr = tipos_count_global.get('CORRETIVA', 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MTBF Global (Horas)", f"{mtbf:.2f}")
        col2.metric("MTTR Global (Horas)", f"{mttr:.2f}")
        col3.metric("% Prev. (Global Frota)", f"{perc_prev:.1f}%")
        col4.metric("% Corr. (Global Frota)", f"{perc_corr:.1f}%")

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
                st.plotly_chart(fig_resp, use_container_width=True)

        # --- MTBF, MTTR E JACK-KNIFE COM RÓTULOS ---
        st.markdown("---")
        st.markdown("### 🚜 Análise Dinâmica: Jack-Knife, MTBF e MTTR")
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
                
                # Rótulos Fixos (text=dimensao) restaurados e visíveis
                fig_jk = px.scatter(dim_stats, x='Falhas', y='MTTR', text=dimensao, size='Downtime',
                                    title=f'Jack-Knife ({dimensao.title()}) - Rótulos Visíveis', opacity=0.8)
                
                fig_jk.update_traces(textposition='top center', textfont=dict(size=11, color='black'))
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

        tab_p1, tab_p2, tab_p3 = st.tabs(["Por Equipamento", "Por Grupo", "Por Subgrupo"])
        with tab_p1: st.plotly_chart(plot_pareto(corretivas, 'EQUIPAMENTO', 'Top 18 - Equipamentos'), use_container_width=True)
        with tab_p2: st.plotly_chart(plot_pareto(corretivas, 'GRUPO', 'Top 18 - Grupos'), use_container_width=True)
        with tab_p3: st.plotly_chart(plot_pareto(corretivas, 'SUBGRUPO', 'Top 18 - Subgrupos'), use_container_width=True)

    else:
        st.info("Faça o upload da planilha de Ordens de Serviço na barra lateral esquerda.")

# =====================================================================
# ABA 2: IA & CONFIABILIDADE AVANÇADA
# =====================================================================
with aba_ia:
    st.header("🧠 Inteligência Artificial & Confiabilidade Avançada")
    if uploaded_file is not None and not corretivas.empty:
        
        # --- 1. MATRIZ DE CRITICIDADE (FMECA) ---
        st.markdown("### 🎯 Matriz de Criticidade (FMECA)")
        fmeca_dim = st.selectbox("Analisar Criticidade por:", ["EQUIPAMENTO", "GRUPO"])
        df_fmeca = corretivas.groupby(fmeca_dim).agg(Falhas=('OS', 'count'), Severidade=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
        df_fmeca['Risco (NPR)'] = df_fmeca['Falhas'] * df_fmeca['Severidade']
        df_fmeca = df_fmeca.sort_values('Risco (NPR)', ascending=False)
        df_fmeca['% Acumulada'] = df_fmeca['Risco (NPR)'].cumsum() / df_fmeca['Risco (NPR)'].sum() * 100
        df_fmeca['Classe'] = np.where(df_fmeca['% Acumulada'] <= 80, 'A (Alta Criticidade)', np.where(df_fmeca['% Acumulada'] <= 95, 'B (Média Criticidade)', 'C (Baixa Criticidade)'))

        fig_fmeca = px.scatter(df_fmeca, x='Falhas', y='Severidade', color='Classe', size='Risco (NPR)', hover_name=fmeca_dim,
                               color_discrete_map={'A (Alta Criticidade)':'red', 'B (Média Criticidade)':'orange', 'C (Baixa Criticidade)':'green'},
                               title=f"Matriz FMECA: Frequência vs Severidade ({fmeca_dim})")
        st.plotly_chart(fig_fmeca, use_container_width=True)

        # --- 2. CURVA DA BANHEIRA E WEIBULL ---
        st.markdown("---")
        st.markdown("### 🛁 Curva da Banheira e Confiabilidade da Frota (Weibull)")

        tbf_data = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
        tbf_data['TBF'] = tbf_data.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
        tbf_clean = tbf_data['TBF'].dropna()
        tbf_clean = tbf_clean[tbf_clean > 0].values

        if len(tbf_clean) > 3:
            shape, loc, scale = st_scipy.weibull_min.fit(tbf_clean, floc=0)
            beta = shape
            eta = scale
            
            estagio = "Mortalidade Infantil (Falhas Prematuras)" if beta < 1 else "Falhas Aleatórias (Vida Útil Normal)" if 1 <= beta <= 1.5 else "Fase de Desgaste (Fim de Vida)"
            st.success(f"**Parâmetro de Forma ($\\beta$):** {beta:.3f} ➔ **Diagnóstico:** {estagio} | **Vida Característica ($\\eta$):** {eta:.2f}h")

            t = np.linspace(0.1, max(tbf_clean) * 1.2, 200)
            reliability = st_scipy.weibull_min.sf(t, shape, loc=0, scale=scale)
            prob_failure = st_scipy.weibull_min.cdf(t, shape, loc=0, scale=scale)
            hazard_rate = st_scipy.weibull_min.pdf(t, shape, loc=0, scale=scale) / reliability
            hazard_rate[np.isinf(hazard_rate)] = 0

            tab_b1, tab_b2, tab_b3, tab_b4 = st.tabs(["Taxa de Falha (Banheira)", "Confiabilidade R(t)", "Probabilidade de Falha F(t)", "RGA"])
            
            with tab_b1:
                fig_haz = go.Figure(go.Scatter(x=t, y=hazard_rate, mode='lines', line=dict(color='orange')))
                fig_haz.update_layout(title="Curva da Banheira: Taxa de Falha h(t)", xaxis_title="Horas", yaxis_title="h(t)")
                st.plotly_chart(fig_haz, use_container_width=True)
            with tab_b2:
                fig_rel = go.Figure(go.Scatter(x=t, y=reliability, mode='lines', line=dict(color='green')))
                fig_rel.update_layout(title="Confiabilidade R(t)", xaxis_title="Horas", yaxis_title="R(t)")
                st.plotly_chart(fig_rel, use_container_width=True)
            with tab_b3:
                fig_prob = go.Figure(go.Scatter(x=t, y=prob_failure, mode='lines', line=dict(color='red')))
                fig_prob.update_layout(title="Probabilidade Acumulada de Falha F(t)", xaxis_title="Horas", yaxis_title="F(t)")
                st.plotly_chart(fig_prob, use_container_width=True)
            with tab_b4:
                tbf_data['Tempo Acumulado'] = tbf_data['TOTAL HORAS DECIMAIS'].cumsum()
                tbf_data['Falhas Acumuladas'] = range(1, len(tbf_data) + 1)
                tbf_data['MTBF Acumulado'] = tbf_data['Tempo Acumulado'] / tbf_data['Falhas Acumuladas']
                fig_rga = px.line(tbf_data, x='Tempo Acumulado', y='MTBF Acumulado', title="RGA - Crescimento da Confiabilidade", markers=True)
                st.plotly_chart(fig_rga, use_container_width=True)
        else:
            st.warning("Dados de TBF insuficientes.")

        # --- 3. MACHINE LEARNING ---
        st.markdown("---")
        st.markdown("### 🤖 Previsão de Quebras (Machine Learning)")
        if not SKLEARN_INSTALLED:
            st.error("Biblioteca `scikit-learn` ausente. Adicione ao requirements.txt.")
        else:
            ml_data = corretivas[['MODELO EQUIPAMENTO', 'EQUIPAMENTO', 'GRUPO', 'RESPONSABILIDADE NÍVEL 1', 'TOTAL HORAS DECIMAIS']].dropna()
            if len(ml_data) > 10:
                rf_model, le_modelo, le_equip, le_grupo, le_resp = treinar_modelo_ml(ml_data.copy())
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                sel_modelo = col_m1.selectbox("Modelo", le_modelo.classes_)
                sel_equip = col_m2.selectbox("Equipamento", le_equip.classes_)
                sel_grupo = col_m3.selectbox("Grupo Afetado", le_grupo.classes_)
                sel_resp = col_m4.selectbox("Responsabilidade", le_resp.classes_)
                
                if st.button("🧠 Prever Downtime"):
                    input_data = np.array([[
                        le_modelo.transform([sel_modelo])[0], le_equip.transform([sel_equip])[0],
                        le_grupo.transform([sel_grupo])[0], le_resp.transform([sel_resp])[0]
                    ]])
                    pred = rf_model.predict(input_data)[0]
                    st.info(f"⏱️ **Downtime Previsto:** A IA estima que uma falha com essas características resultará em **{pred:.2f} horas** paradas.")
            else:
                st.warning("Histórico insuficiente para treinar IA (min: 10 corretivas).")

# =====================================================================
# ABA 3: PLANO DE AÇÃO 5W2H (Google Sheets)
# =====================================================================
with aba_plano_acao:
    st.header("📋 Plano de Ação 5W2H")
    url_planilha = "https://docs.google.com/spreadsheets/d/1mrfp_qDdX5_6sJVT5-Gk3NzM3nKqznmlR6rwDsXZN2s/edit?gid=0#gid=0"
    conn = st.connection("gsheets", type=GSheetsConnection)
    colunas_5w2h = ["NOME DO CONTRATO", "What?", "Why?", "Where?", "When?", "Who?", "How?", "How Much?", "Status"]

    try:
        df_acao = conn.read(spreadsheet=url_planilha)
        if df_acao.empty or len(df_acao.columns) < 2:
            df_acao = pd.DataFrame(columns=colunas_5w2h)
            df_acao.loc[0] = [""] * len(colunas_5w2h)
    except Exception:
        df_acao = pd.DataFrame(columns=colunas_5w2h)
        df_acao.loc[0] = [""] * len(colunas_5w2h)

    df_editado = st.data_editor(df_acao, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Salvar no Google Sheets"):
        with st.spinner("Salvando..."):
            try:
                conn.update(spreadsheet=url_planilha, data=df_editado)
                st.success("Dados salvos no Google Sheets com sucesso!")
            except Exception:
                st.warning("⚠️ **Bloqueio do Google:** Para salvar direto na nuvem, você deve gerar um JSON de Service Account no Google Cloud e colocar nas configurações (Secrets) do seu app Streamlit. Como fallback, utilize o botão abaixo para baixar o CSV.")
                
    csv = df_editado.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Plano de Ação (CSV Local)", data=csv, file_name='plano_acao_backup.csv', mime='text/csv')

# =====================================================================
# ABA 4: CONTROLE DE COMPONENTES E LDA
# =====================================================================
with aba_lda:
    st.header("🛠️ Análise de Dados de Vida (LDA) - Componentes")
    
    if not LIFELINES_INSTALLED:
        st.error("⚠️ Biblioteca `lifelines` ausente no requirements.txt.")
    else:
        file_lda = st.file_uploader("Carregue a planilha de Controle de Componentes", type=["xlsx"], key="lda_uploader")
        
        if file_lda is not None:
            df_comp = pd.read_excel(file_lda, sheet_name=0)
            df_comp.columns = df_comp.columns.str.replace('\n', ' ').str.replace('  ', ' ').str.strip()
            
            if 'SITUAÇÃO DO COMPONENTE' in df_comp.columns and 'HORAS TRABALHADAS DO COMPONENTE' in df_comp.columns:
                
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
                    st.write(f"**Amostras:** {len(df_alvo)} | **Falhas:** {falhas_count} | **Censuras:** {susp_count}")
                    
                    if falhas_count > 0:
                        wf = WeibullFitter()
                        wf.fit(df_alvo['Horas_LDA'], event_observed=df_alvo['Status_LDA'])
                        mttf_lda = wf.lambda_ * gamma(1 + (1/wf.rho_))
                        st.success(f"**Parâmetros Weibull:** $\\beta$ = {wf.rho_:.2f} | $\\eta$ = {wf.lambda_:.2f}h | **MTTF Estimado:** {mttf_lda:.2f}h")
                    else:
                        st.warning("Sem falhas confirmadas. Curvas avançadas não podem ser geradas.")
                
                # --- MAPA DE CALOR SEGURO COM CACHE ---
                st.markdown("---")
                st.markdown("### 🔥 Mapa de Calor: Vida Útil Restante da Frota")
                
                # Chama a função cachead para evitar lentidão
                component_mttf = calcular_mttf_componentes(df_comp)

                df_ativos = df_comp[(df_comp['Status_LDA'] == 0) & (df_comp['Horas_LDA'] > 0)].copy()
                df_ativos['MTTF'] = df_ativos['COMPONENTE'].map(component_mttf)
                df_ativos = df_ativos.dropna(subset=['MTTF'])

                if not df_ativos.empty and 'TAG' in df_ativos.columns:
                    df_ativos['Vida_Consumida_%'] = (df_ativos['Horas_LDA'] / df_ativos['MTTF']) * 100
                    df_ativos['EQUIP'] = df_ativos['TAG'].astype(str).apply(lambda x: x.split(' ')[0])
                    
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        max_vida = float(df_ativos['Vida_Consumida_%'].max()) if not df_ativos.empty else 100.0
                        faixa_vida = st.slider("Filtro de % Vida Consumida:", min_value=0.0, max_value=max(200.0, max_vida), value=(0.0, max(100.0, max_vida)))
                    with col_f2:
                        comps_heatmap = df_ativos['COMPONENTE'].unique().tolist()
                        selecao_comps = st.multiselect("Selecione Componentes:", comps_heatmap, default=comps_heatmap)

                    df_heat_filtered = df_ativos[(df_ativos['Vida_Consumida_%'] >= faixa_vida[0]) & (df_ativos['Vida_Consumida_%'] <= faixa_vida[1]) & (df_ativos['COMPONENTE'].isin(selecao_comps))]

                    if not df_heat_filtered.empty:
                        heatmap_data = df_heat_filtered.pivot_table(index='EQUIP', columns='COMPONENTE', values='Vida_Consumida_%', aggfunc='mean').fillna(0)
                        fig_heat = px.imshow(heatmap_data, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r", title="Porcentagem (%) do MTTF Consumida")
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                        st.markdown("#### 🕒 Tabela de Vida Útil (MTTF Estimado)")
                        df_mttf_display = pd.DataFrame(list(component_mttf.items()), columns=['Componente', 'MTTF (Horas)'])
                        df_mttf_display = df_mttf_display[df_mttf_display['Componente'].isin(selecao_comps)].sort_values('MTTF (Horas)')
                        df_mttf_display['MTTF (Horas)'] = df_mttf_display['MTTF (Horas)'].round(2)
                        st.dataframe(df_mttf_display, use_container_width=True)
                    else:
                        st.warning("Nenhum ativo operando atende a esses filtros.")
                else:
                    st.info("Para gerar o mapa de calor é necessário que existam peças com falhas (para calcular o MTTF) convivendo com peças ativas.")
            else:
                st.error("A planilha não possui as colunas 'SITUAÇÃO DO COMPONENTE' e/ou 'HORAS TRABALHADAS DO COMPONENTE'.")
