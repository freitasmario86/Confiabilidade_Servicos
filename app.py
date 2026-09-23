import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st_scipy
from scipy.stats import poisson
from scipy.special import gamma
import io
import math
from datetime import datetime
import warnings
import unicodedata
import requests
from streamlit_gsheets import GSheetsConnection

try:
    from fpdf import FPDF
    FPDF_INSTALLED = True
except ImportError:
    FPDF_INSTALLED = False

try:
    from lifelines import WeibullFitter, KaplanMeierFitter
    LIFELINES_INSTALLED = True
except ImportError:
    LIFELINES_INSTALLED = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard de Manutenção CIM", layout="wide", page_icon="⚙️")

# =====================================================================
# FUNÇÕES DE CACHE & UTILIDADES
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
def calcular_metricas_componentes(df_comp, alpha=0.05):
    component_metrics = {}
    if not LIFELINES_INSTALLED:
        return component_metrics
    for comp in df_comp['COMPONENTE'].dropna().unique():
        df_c = df_comp[(df_comp['COMPONENTE'] == comp) & (df_comp['Horas_LDA'] > 0)]
        if df_c['Status_LDA'].sum() > 0:
            try:
                wf_heat = WeibullFitter(alpha=alpha)
                wf_heat.fit(df_c['Horas_LDA'], event_observed=df_c['Status_LDA'])
                mttf = wf_heat.lambda_ * gamma(1 + (1/wf_heat.rho_))
                # Cálculo da probabilidade de falha F(t) no ponto do MTTF
                prob_mttf = (1 - np.exp(- (mttf / wf_heat.lambda_) ** wf_heat.rho_)) * 100
                component_metrics[comp] = {
                    'MTTF': mttf, 
                    'Prob_MTTF': prob_mttf
                }
            except Exception:
                pass
    return component_metrics

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_pdf_lines(pdf, text, width):
    if pd.isna(text) or str(text).strip() == "" or str(text) in ["nan", "None", "<NA>"]:
        return 1
    text = str(text)
    lines = 0
    for paragraph in text.split('\n'):
        lines += max(1, math.ceil(pdf.get_string_width(paragraph) / (width - 2)))
    return lines

def format_date_safe(val):
    try:
        if pd.isnull(val) or str(val).strip() in ['NaT', 'nan', 'None', '', '<NA>']:
            return ""
        return pd.to_datetime(val).strftime('%d/%m/%Y')
    except:
        return ""

# --- NAVEGAÇÃO POR ABAS PRINCIPAIS ---
aba_dashboard, aba_ia, aba_estrategia, aba_plano_acao, aba_lda = st.tabs([
    "📊 Dashboard de OS", 
    "🧠 IA & Confiabilidade Avançada",
    "⏱️ 5 Etapas & Estratégia",
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
        st.sidebar.header("Filtros em Cascata")
        
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
        disp_inerente = (mtbf / (mtbf + mttr)) * 100 if (mtbf + mttr) > 0 else 0
        
        tipos_count_global = df['TIPO'].value_counts(normalize=True) * 100
        perc_prev = tipos_count_global.get('PREVENTIVA', 0)
        perc_corr = tipos_count_global.get('CORRETIVA', 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MTBF Global (Horas)", f"{mtbf:.2f}")
        col2.metric("MTTR Global (Horas)", f"{mttr:.2f}")
        col3.metric("Disponibilidade (Ai)", f"{disp_inerente:.1f}%")
        col4.metric("% Prev. (Global Frota)", f"{perc_prev:.1f}%")
        col5.metric("% Corr. (Global Frota)", f"{perc_corr:.1f}%")

        st.markdown("---")
        col_evol, col_resp = st.columns([2, 1])
        with col_evol:
            st.markdown("### 📅 Evolução Mensal Histórica")
            corretivas_evol = df_cascaded[df_cascaded['TIPO'] == 'CORRETIVA']
            if not corretivas_evol.empty:
                corretivas_evol['Ano-Mês'] = corretivas_evol['DATA INÍCIO'].dt.strftime('%Y-%m')
                evol_stats = corretivas_evol.groupby('Ano-Mês').agg(Falhas=('EQUIPAMENTO', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
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

        st.markdown("---")
        st.markdown("### 🚜 Análise Dinâmica: Jack-Knife, MTBF e MTTR")
        dimensao = st.radio("Selecione a Dimensão de Análise:", ["EQUIPAMENTO", "GRUPO", "SUBGRUPO"], horizontal=True)
        
        if not corretivas.empty and dimensao in corretivas.columns:
            dim_stats = corretivas.groupby(dimensao).agg(Falhas=('EQUIPAMENTO', 'count'), Downtime=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
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
                
                fig_jk = px.scatter(dim_stats, x='Falhas', y='MTTR', text=dimensao, size='Downtime', title=f'Jack-Knife ({dimensao.title()})', opacity=0.8)
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
                df_mtbf_sorted = dim_stats.sort_values('MTBF', ascending=False)
                fig_mtbf = px.bar(df_mtbf_sorted, x=dimensao, y='MTBF', text=df_mtbf_sorted['MTBF'].round(1), title=f"MTBF por {dimensao.title()}")
                fig_mtbf.update_traces(textposition='outside')
                st.plotly_chart(fig_mtbf, use_container_width=True)
                
            with tab_eq3:
                df_mttr_sorted = dim_stats.sort_values('MTTR', ascending=False)
                fig_mttr = px.bar(df_mttr_sorted, x=dimensao, y='MTTR', text=df_mttr_sorted['MTTR'].round(1), color_discrete_sequence=['indianred'], title=f"MTTR por {dimensao.title()}")
                fig_mttr.update_traces(textposition='outside')
                st.plotly_chart(fig_mttr, use_container_width=True)

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

        # --- NOVA SEÇÃO: FALHAS REPETIDAS (BAD ACTORS) ---
        st.markdown("---")
        st.markdown("### 🔄 Falhas Repetidas por Equipamento (Bad Actors)")
        
        if not corretivas.empty:
            df_repetidas = corretivas.groupby('EQUIPAMENTO').agg(
                Qtd_Falhas=('EQUIPAMENTO', 'count'), 
                Downtime=('TOTAL HORAS DECIMAIS', 'sum')
            ).reset_index()
            
            # Filtra apenas equipamentos que falharam mais de 1 vez e ordena
            df_repetidas = df_repetidas[df_repetidas['Qtd_Falhas'] > 1].sort_values('Qtd_Falhas', ascending=False)
            
            if not df_repetidas.empty:
                df_repetidas['MTBF (Horas)'] = ((dias_operacao * 24) - df_repetidas['Downtime']) / df_repetidas['Qtd_Falhas']
                df_repetidas['MTBF (Horas)'] = df_repetidas['MTBF (Horas)'].apply(lambda x: max(x, 0)).round(2)
                df_repetidas['Downtime (Horas)'] = df_repetidas['Downtime'].round(2)
                
                df_exibicao = df_repetidas[['EQUIPAMENTO', 'Qtd_Falhas', 'Downtime (Horas)', 'MTBF (Horas)']].head(15)
                
                col_bad1, col_bad2 = st.columns([1, 2])
                with col_bad1:
                    st.dataframe(
                        df_exibicao.rename(columns={'Qtd_Falhas': 'Nº de Falhas'}),
                        use_container_width=True,
                        hide_index=True
                    )
                with col_bad2:
                    fig_bad = px.bar(
                        df_exibicao, 
                        x='EQUIPAMENTO', 
                        y='Qtd_Falhas', 
                        text='Qtd_Falhas',
                        hover_data=['Downtime (Horas)', 'MTBF (Horas)'],
                        title="Top 15 Equipamentos com Falhas Repetidas",
                        color='Qtd_Falhas',
                        color_continuous_scale='Reds'
                    )
                    fig_bad.update_traces(textposition='outside')
                    fig_bad.update_layout(yaxis_title="Número de Falhas")
                    st.plotly_chart(fig_bad, use_container_width=True)
            else:
                st.success("🎉 Nenhum equipamento apresentou falhas repetidas no período filtrado!")

    else:
        st.info("Faça o upload da Planilha de Ordens de Serviço (OS).")

# =====================================================================
# ABA 2: IA & CONFIABILIDADE AVANÇADA
# =====================================================================
with aba_ia:
    st.header("🧠 Inteligência Artificial & Confiabilidade Avançada")
    if 'corretivas' in locals() and not corretivas.empty:
        
        st.markdown("### 🎯 Matriz de Criticidade (FMECA)")
        fmeca_dim = st.selectbox("Analisar Criticidade por:", ["EQUIPAMENTO", "GRUPO", "SUBGRUPO"], key='fmeca')
        df_fmeca = corretivas.groupby(fmeca_dim).agg(Falhas=('EQUIPAMENTO', 'count'), Severidade=('TOTAL HORAS DECIMAIS', 'sum')).reset_index()
        df_fmeca['Risco (NPR)'] = df_fmeca['Falhas'] * df_fmeca['Severidade']
        df_fmeca = df_fmeca.sort_values('Risco (NPR)', ascending=False)
        df_fmeca['% Acumulada'] = df_fmeca['Risco (NPR)'].cumsum() / df_fmeca['Risco (NPR)'].sum() * 100
        df_fmeca['Classe'] = np.where(df_fmeca['% Acumulada'] <= 80, 'A (Alta Criticidade)', np.where(df_fmeca['% Acumulada'] <= 95, 'B (Média Criticidade)', 'C (Baixa Criticidade)'))

        fig_fmeca = px.scatter(df_fmeca, x='Falhas', y='Severidade', color='Classe', size='Risco (NPR)', text=fmeca_dim, hover_name=fmeca_dim,
                               color_discrete_map={'A (Alta Criticidade)':'red', 'B (Média Criticidade)':'orange', 'C (Baixa Criticidade)':'green'},
                               title=f"Matriz FMECA: Frequência vs Severidade ({fmeca_dim})")
        fig_fmeca.update_traces(textposition='top center', textfont=dict(size=10, color='black'), cliponaxis=False)
        st.plotly_chart(fig_fmeca, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🛁 Curva da Banheira (Diagnóstico de Frota) e Distribuições")
        tbf_data_global = corretivas.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
        tbf_data_global['TBF'] = tbf_data_global.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
        tbf_clean_global = tbf_data_global['TBF'].dropna()
        tbf_clean_global = tbf_clean_global[tbf_clean_global > 0].values

        if len(tbf_clean_global) > 3:
            # 1. Encontrar a melhor distribuição (Weibull, Exponencial, Lognormal) via AIC
            # Weibull
            shape_w, loc_w, scale_w = st_scipy.weibull_min.fit(tbf_clean_global, floc=0)
            ll_w = np.sum(st_scipy.weibull_min.logpdf(tbf_clean_global, shape_w, loc=loc_w, scale=scale_w))
            aic_w = 4 - 2 * ll_w
            
            # Exponencial
            loc_e, scale_e = st_scipy.expon.fit(tbf_clean_global, floc=0)
            ll_e = np.sum(st_scipy.expon.logpdf(tbf_clean_global, loc=loc_e, scale=scale_e))
            aic_e = 2 - 2 * ll_e
            
            # Lognormal
            shape_l, loc_l, scale_l = st_scipy.lognorm.fit(tbf_clean_global, floc=0)
            ll_l = np.sum(st_scipy.lognorm.logpdf(tbf_clean_global, shape_l, loc=loc_l, scale=scale_l))
            aic_l = 4 - 2 * ll_l
            
            resultados_dist = [
                {'nome': 'Weibull', 'aic': aic_w, 'params': {'shape': shape_w, 'scale': scale_w}, 'obj': st_scipy.weibull_min},
                {'nome': 'Exponencial', 'aic': aic_e, 'params': {'scale': scale_e}, 'obj': st_scipy.expon},
                {'nome': 'Lognormal', 'aic': aic_l, 'params': {'shape': shape_l, 'scale': scale_l}, 'obj': st_scipy.lognorm}
            ]
            melhor_dist = min(resultados_dist, key=lambda x: x['aic'])
            
            st.success(f"🏆 **Melhor Distribuição Ajustada:** {melhor_dist['nome']} (AIC: {melhor_dist['aic']:.1f})")
            
            # Diagnóstico da Curva da Banheira
            estagio = "Mortalidade Infantil (Falhas Prematuras)" if shape_w < 1 else "Falhas Aleatórias (Vida Útil Normal)" if 1 <= shape_w <= 1.5 else "Fase de Desgaste (Fim de Vida)"
            st.info(f"💡 **Diagnóstico da Frota (Baseado em Weibull $\\beta={shape_w:.3f}$):** {estagio}")

            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.markdown("#### Probabilidade e Confiabilidade")
                t_input = st.number_input("Insira o Tempo Operacional $t$ (Horas) para cálculo:", min_value=1.0, value=float(round(np.mean(tbf_clean_global))), step=10.0)
                
                if melhor_dist['nome'] == 'Weibull':
                    rt_val = st_scipy.weibull_min.sf(t_input, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale']) * 100
                    ft_val = 100 - rt_val
                    st.write(f"**Parâmetros:** $\\beta$ (Forma) = {melhor_dist['params']['shape']:.3f} | $\\eta$ (Escala) = {melhor_dist['params']['scale']:.1f}h")
                elif melhor_dist['nome'] == 'Exponencial':
                    rt_val = st_scipy.expon.sf(t_input, loc=0, scale=melhor_dist['params']['scale']) * 100
                    ft_val = 100 - rt_val
                    st.write(f"**Parâmetros:** MTBF (Escala) = {melhor_dist['params']['scale']:.1f}h | $\\lambda$ = {1/melhor_dist['params']['scale']:.6f}")
                else:
                    rt_val = st_scipy.lognorm.sf(t_input, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale']) * 100
                    ft_val = 100 - rt_val
                    st.write(f"**Parâmetros:** $\\sigma$ (Forma) = {melhor_dist['params']['shape']:.3f} | $\\mu$ (Escala/exp) = {melhor_dist['params']['scale']:.1f}h")

                st.metric("Confiabilidade R(t)", f"{rt_val:.1f}%")
                st.metric("Probabilidade de Falha F(t)", f"{ft_val:.1f}%")

            with col_dist2:
                t_g = np.linspace(0.1, max(tbf_clean_global) * 1.2, 200)
                if melhor_dist['nome'] == 'Weibull':
                    reliab = st_scipy.weibull_min.sf(t_g, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale'])
                    pdf_vals = st_scipy.weibull_min.pdf(t_g, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale'])
                elif melhor_dist['nome'] == 'Exponencial':
                    reliab = st_scipy.expon.sf(t_g, loc=0, scale=melhor_dist['params']['scale'])
                    pdf_vals = st_scipy.expon.pdf(t_g, loc=0, scale=melhor_dist['params']['scale'])
                else:
                    reliab = st_scipy.lognorm.sf(t_g, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale'])
                    pdf_vals = st_scipy.lognorm.pdf(t_g, melhor_dist['params']['shape'], loc=0, scale=melhor_dist['params']['scale'])

                hazard_rate_g = pdf_vals / reliab
                hazard_rate_g[np.isinf(hazard_rate_g)] = 0
                
                perc_reliab = reliab * 100
                perc_fail = (1 - reliab) * 100

                tab_h, tab_prob = st.tabs(["Taxa de Falha h(t)", "Curvas de Probabilidade e Confiabilidade"])
                
                with tab_h:
                    fig_haz_g = go.Figure(go.Scatter(x=t_g, y=hazard_rate_g, mode='lines', line=dict(color='orange')))
                    fig_haz_g.update_layout(title=f"Curva da Banheira: Taxa de Falha h(t) - {melhor_dist['nome']}", xaxis_title="Horas Operacionais", yaxis_title="h(t)")
                    st.plotly_chart(fig_haz_g, use_container_width=True)
                
                with tab_prob:
                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Scatter(x=t_g, y=perc_reliab, mode='lines', name='Confiabilidade R(t)', line=dict(color='blue')))
                    fig_prob.add_trace(go.Scatter(x=t_g, y=perc_fail, mode='lines', name='Prob. Falha F(t)', line=dict(color='red')))
                    fig_prob.update_layout(title=f"R(t) e F(t) - {melhor_dist['nome']}", xaxis_title="Horas Operacionais", yaxis_title="Porcentagem (%)")
                    st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📈 RGA - Reliability Growth Analysis (Modelo Crow-AMSAA)")
            
            df_rga = corretivas.sort_values(by='DATA INÍCIO').dropna(subset=['DATA INÍCIO'])
            if len(df_rga) > 3:
                data_inicial = df_rga['DATA INÍCIO'].min()
                df_rga['Tempo_Acumulado_Horas'] = (df_rga['DATA INÍCIO'] - data_inicial).dt.total_seconds() / 3600
                df_rga = df_rga[df_rga['Tempo_Acumulado_Horas'] > 0]
                df_rga['Falhas_Acumuladas'] = np.arange(1, len(df_rga) + 1)
                
                if len(df_rga) > 2:
                    x_log = np.log(df_rga['Tempo_Acumulado_Horas'])
                    y_log = np.log(df_rga['Falhas_Acumuladas'])
                    beta_rga, log_lambda_rga = np.polyfit(x_log, y_log, 1)
                    lambda_rga = np.exp(log_lambda_rga)
                    
                    col_rga1, col_rga2 = st.columns([1, 2])
                    
                    with col_rga1:
                        st.write(f"**Parâmetro de Crescimento ($\\beta$):** {beta_rga:.3f}")
                        st.write(f"**Escala ($\\lambda$):** {lambda_rga:.4f}")
                        if beta_rga < 1:
                            st.success("✅ **Confiabilidade Crescente!** A taxa de falhas está diminuindo. Melhorias sendo efetivas.")
                        elif beta_rga > 1.1:
                            st.error("🚨 **Confiabilidade Degradando!** A taxa de falhas está aumentando. Revisar planos de manutenção.")
                        else:
                            st.info("⚖️ **Confiabilidade Estável.** Taxa de falhas constante (comportamento exponencial).")
                            
                    with col_rga2:
                        fig_rga = go.Figure()
                        fig_rga.add_trace(go.Scatter(x=df_rga['Tempo_Acumulado_Horas'], y=df_rga['Falhas_Acumuladas'], mode='markers', name='Dados Reais', marker=dict(color='black')))
                        
                        t_trend = np.linspace(min(df_rga['Tempo_Acumulado_Horas']), max(df_rga['Tempo_Acumulado_Horas']), 100)
                        n_trend = lambda_rga * (t_trend ** beta_rga)
                        fig_rga.add_trace(go.Scatter(x=t_trend, y=n_trend, mode='lines', name='Curva Crow-AMSAA', line=dict(dash='dash', color='red')))
                        
                        fig_rga.update_layout(title="Crescimento de Confiabilidade (Log-Log)", xaxis_title="Tempo Acumulado (Horas)", yaxis_title="Falhas Acumuladas N(t)", xaxis_type="log", yaxis_type="log")
                        st.plotly_chart(fig_rga, use_container_width=True)
            else:
                st.warning("Falhas insuficientes para Análise RGA.")
        else:
            st.warning("Dados de TBF insuficientes.")
    else:
        st.info("Carregue a planilha na aba principal.")

# =====================================================================
# ABA 3: 5 ETAPAS E ESTRATÉGIA
# =====================================================================
with aba_estrategia:
    st.header("⏱️ Estratégia de Reparo, Inspeções e Sobressalentes")
    
    if 'corretivas' in locals() and not corretivas.empty:
        st.markdown("### 📦 1. Previsão de Sobressalentes (Spare Parts Forecasting)")
        df_sp = corretivas.copy()
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        mod_sp = col_s1.selectbox("Filtre o Modelo:", ["Todos"] + df_sp['MODELO EQUIPAMENTO'].dropna().unique().tolist())
        if mod_sp != "Todos": df_sp = df_sp[df_sp['MODELO EQUIPAMENTO'] == mod_sp]
        
        eq_sp = col_s2.selectbox("Filtre o Equipamento:", ["Todos"] + df_sp['EQUIPAMENTO'].dropna().unique().tolist())
        if eq_sp != "Todos": df_sp = df_sp[df_sp['EQUIPAMENTO'] == eq_sp]
        
        gr_sp = col_s3.selectbox("Filtre o Grupo:", ["Todos"] + df_sp['GRUPO'].dropna().unique().tolist())
        if gr_sp != "Todos": df_sp = df_sp[df_sp['GRUPO'] == gr_sp]
        
        sub_sp = col_s4.selectbox("Selecione o Subgrupo Alvo:", ["Todos"] + df_sp['SUBGRUPO'].dropna().unique().tolist())
        if sub_sp != "Todos": df_sp = df_sp[df_sp['SUBGRUPO'] == sub_sp]
        
        horas_projecao = st.number_input("Insira as Horas de Projeção (ex: 4380h para 6 meses)", value=4380, min_value=1)
        
        falhas_spare = len(df_sp)
        if falhas_spare > 0 and (mod_sp != "Todos" or eq_sp != "Todos" or gr_sp != "Todos" or sub_sp != "Todos"):
            df_pop = df_filtered.copy()
            if mod_sp != "Todos": df_pop = df_pop[df_pop['MODELO EQUIPAMENTO'] == mod_sp]
            if eq_sp != "Todos": df_pop = df_pop[df_pop['EQUIPAMENTO'] == eq_sp]
            
            qtd_ativos_spare = df_pop['EQUIPAMENTO'].nunique() if not df_pop.empty else 1
            mtbf_spare = ((dias_operacao * 24 * qtd_ativos_spare) - df_sp['TOTAL HORAS DECIMAIS'].sum()) / falhas_spare
            mtbf_spare = mtbf_spare if mtbf_spare > 0 else 1
            
            lambda_expected = (horas_projecao / mtbf_spare) * qtd_ativos_spare
            estoque_recomendado = poisson.ppf(0.95, lambda_expected)
            st.success(f"🛒 **Estoque Recomendado (95% Segurança): {int(estoque_recomendado)} unidades** (Demanda Média: {lambda_expected:.1f} falhas)")
        else:
            st.warning("Selecione os filtros acima até chegar no item alvo.")

        st.markdown("---")
        st.markdown("### 🔍 2. Cálculo de Intervalo de Inspeções (F(t) e MTBF)")
        dim_insp = st.selectbox("Nível para Intervalo de Inspeção:", ["MODELO EQUIPAMENTO", "EQUIPAMENTO", "GRUPO"], key='insp')
        opcoes_insp = corretivas[dim_insp].dropna().unique().tolist()
        alvo_insp = st.selectbox("Selecione o Alvo para Inspeção:", opcoes_insp, key='insp_alvo')
        
        df_insp = corretivas[corretivas[dim_insp] == alvo_insp]
        tbf_insp = df_insp.sort_values(by=['EQUIPAMENTO', 'DATA INÍCIO'])
        tbf_insp['TBF'] = tbf_insp.groupby('EQUIPAMENTO')['DATA INÍCIO'].diff().dt.total_seconds() / 3600
        tbf_clean_insp = tbf_insp['TBF'].dropna()
        tbf_clean_insp = tbf_clean_insp[tbf_clean_insp > 0].values
        
        if len(tbf_clean_insp) > 3:
            shape_i, loc_i, scale_i = st_scipy.weibull_min.fit(tbf_clean_insp, floc=0)
            B2 = st_scipy.weibull_min.ppf(0.02, shape_i, scale=scale_i)
            B10 = st_scipy.weibull_min.ppf(0.10, shape_i, scale=scale_i)
            mtbf_i = scale_i * gamma(1 + (1/shape_i))
            
            st.write(f"**Memorial Estatístico:** $\\beta$ = {shape_i:.3f} | $\\eta$ = {scale_i:.1f}h | **B2 = {B2:.1f}h** | **B10 = {B10:.1f}h** | **MTBF = {mtbf_i:.1f}h**")
            
            col_tabela, col_grafico = st.columns([1, 2])
            with col_tabela:
                st.markdown("<br>", unsafe_allow_html=True)
                tabela_inspecao = pd.DataFrame({"Criticidade": ["S/Q (Segurança)", "A", "B", "C"], "Cálculo": ["B2 / 3", "B10 / 3", "MTBF / 2", "Histórico"], "Inspeção (h)": [f"{B2/3:.0f}", f"{B10/3:.0f}", f"{mtbf_i/2:.0f}", "-"]})
                st.table(tabela_inspecao)
            
            with col_grafico:
                tab_cdf, tab_log = st.tabs(["CDF Linear (F(t))", "Probability Plot (Log-Log)"])
                
                t_plot = np.linspace(0.1, max(tbf_clean_insp)*1.5, 300)
                cdf_plot = st_scipy.weibull_min.cdf(t_plot, shape_i, scale=scale_i) * 100
                
                kmf_i = KaplanMeierFitter()
                kmf_i.fit(tbf_clean_insp, event_observed=np.ones_like(tbf_clean_insp))
                emp_f = (1 - kmf_i.survival_function_['KM_estimate']) * 100

                with tab_cdf:
                    fig_cdf = go.Figure()
                    fig_cdf.add_trace(go.Scatter(x=t_plot, y=cdf_plot, mode='lines', name='CDF Teórica (Weibull)', line=dict(color='blue')))
                    fig_cdf.add_trace(go.Scatter(x=emp_f.index, y=emp_f.values, mode='markers', name='CDF Empírica', marker=dict(color='black', symbol='circle')))
                    
                    fig_cdf.add_vline(x=B2, line_dash="dash", line_color="red", annotation_text="B2 (2%)", annotation_position="top left")
                    fig_cdf.add_vline(x=B10, line_dash="dash", line_color="orange", annotation_text="B10 (10%)", annotation_position="top right")
                    fig_cdf.add_vline(x=mtbf_i, line_dash="dash", line_color="black", annotation_text="MTBF", annotation_position="bottom right")
                    fig_cdf.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
                    fig_cdf.add_hline(y=10, line_dash="dot", line_color="orange", opacity=0.5)
                    fig_cdf.update_layout(title="Curva de Probabilidade Acumulada - F(t)", xaxis_title="Horas", yaxis_title="Fração de Falhas (%)", yaxis=dict(range=[0, max(20, (st_scipy.weibull_min.cdf(mtbf_i, shape_i, scale=scale_i)*100)+15)]))
                    st.plotly_chart(fig_cdf, use_container_width=True)

                with tab_log:
                    t_log = t_plot[t_plot > 0]
                    cdf_log = st_scipy.weibull_min.cdf(t_log, shape_i, scale=scale_i)
                    
                    cdf_log = np.where(cdf_log == 0, 1e-5, cdf_log)
                    cdf_log = np.where(cdf_log == 1, 0.9999, cdf_log)
                    
                    fig_log = go.Figure()
                    fig_log.add_trace(go.Scatter(x=t_log, y=cdf_log*100, mode='lines', name='Linha de Tendência', line=dict(color='blue')))
                    fig_log.add_trace(go.Scatter(x=emp_f.index, y=emp_f.values, mode='markers', name='Dados', marker=dict(color='black')))
                    
                    fig_log.update_layout(
                        title="Weibull Probability Plot",
                        xaxis_title="Tempo (Horas)",
                        yaxis_title="Probabilidade Cumulativa (%)",
                        xaxis_type="log",
                        yaxis=dict(
                            type="log",
                            tickvals=[1, 2, 5, 10, 20, 50, 90, 99],
                            ticktext=["1", "2", "5", "10", "20", "50", "90", "99"]
                        )
                    )
                    st.plotly_chart(fig_log, use_container_width=True)
        else:
            st.warning("Dados insuficientes para calcular os parâmetros B2 e B10.")

        st.markdown("---")
        st.markdown("### 🛠️ 3. As 5 Etapas do Tempo Ótimo de Reparo (ORT)")
        
        if len(tbf_clean_insp) > 3:
            if shape_i <= 1:
                st.error(f"$\\beta$ ({shape_i:.3f}) $\le$ 1. A manutenção baseada no tempo não é econômica para este modo de falha.")
            else:
                col_c1, col_c2 = st.columns(2)
                C_pm = col_c1.number_input("Custo de Manutenção Preventiva (Cpm)", value=1500)
                C_cm = col_c2.number_input("Custo de Manutenção Corretiva (Cmc)", value=8000)
                
                if C_cm > C_pm:
                    t_opt = np.linspace(1, max(tbf_clean_insp)*1.5, 500)
                    dt = t_opt[1] - t_opt[0]
                    R_t = st_scipy.weibull_min.sf(t_opt, shape_i, loc=0, scale=scale_i)
                    F_t = 1 - R_t
                    integral_R = np.cumsum(R_t) * dt
                    C_t = (C_pm * R_t + C_cm * F_t) / integral_R
                    min_idx = np.argmin(C_t)
                    optimal_time = t_opt[min_idx]
                    
                    fig_ort = go.Figure(go.Scatter(x=t_opt, y=C_t, mode='lines', line=dict(color='blue')))
                    fig_ort.add_vline(x=optimal_time, line_dash="dash", line_color="red", annotation_text=f"Tempo Ótimo: {optimal_time:.0f}h")
                    fig_ort.update_layout(title="Optimal Replacement Time Estimation (CPUT)", xaxis_title="Replacement time (h)", yaxis_title="Cost per unit time")
                    st.plotly_chart(fig_ort, use_container_width=True)
        else:
            st.warning("Selecione um alvo no filtro de Inspeções com histórico suficiente.")
    else:
        st.info("Carregue a planilha na aba principal.")

# =====================================================================
# ABA 4: PLANO DE AÇÃO 5W2H (Integração e Auditoria)
# =====================================================================
with aba_plano_acao:
    st.header("📋 Plano de Ação 5W2H")
    
    URL_APPS_SCRIPT = "https://script.google.com/macros/s/AKfycbwQu0k_oQdCCSvlhz9icieN5xTyk6FQLwdk2IVUWgCEWQsDH6nRyUGl9u7qe-BNtuib7A/exec"
    url_planilha = "https://docs.google.com/spreadsheets/d/1mrfp_qDdX5_6sJVT5-Gk3NzM3nKqznmlR6rwDsXZN2s/edit?gid=0#gid=0"
    
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    colunas_5w2h = [
        "NOME DO CONTRATO", "What? (O que será feito)", "Why? (Por que)", 
        "Where? (Onde/Equipamento)", "When? (Prazo)", "Prazo Original", 
        "Who? (Responsável)", "How? (Como)", "How Much? (Custo Estimado)", 
        "Status", "Motivo", "Nova Data", "Dias de Atraso", "Histórico"
    ]
    
    try:
        df_acao = conn.read(spreadsheet=url_planilha)
        for col in colunas_5w2h:
            if col not in df_acao.columns:
                df_acao[col] = ""
        df_acao = df_acao[colunas_5w2h]
    except Exception:
        df_acao = pd.DataFrame(columns=colunas_5w2h)
        df_acao.loc[0] = [""] * len(colunas_5w2h)

    for col in colunas_5w2h:
        if col not in ['When? (Prazo)', 'Prazo Original', 'Nova Data']:
            df_acao[col] = df_acao[col].astype(str).replace({'nan': '', 'None': '', '<NA>': '', 'NaN': ''})

    df_acao['How Much? (Custo Estimado)'] = pd.to_numeric(df_acao['How Much? (Custo Estimado)'], errors='coerce')
    
    df_acao['When? (Prazo)'] = pd.to_datetime(df_acao['When? (Prazo)'], format='%d/%m/%Y', errors='coerce').dt.date
    df_acao['Prazo Original'] = pd.to_datetime(df_acao['Prazo Original'], format='%d/%m/%Y', errors='coerce').dt.date
    df_acao['Nova Data'] = pd.to_datetime(df_acao['Nova Data'], format='%d/%m/%Y', errors='coerce').dt.date

    st.markdown("#### Filtros do Plano de Ação")
    col_f1, col_f2 = st.columns(2)
    contratos_disp = df_acao["NOME DO CONTRATO"].dropna().astype(str).unique().tolist()
    status_disp = df_acao["Status"].dropna().astype(str).unique().tolist()
    
    filtro_contrato = col_f1.multiselect("Filtrar por Contrato:", [c for c in contratos_disp if c not in ["", "nan", "None", "<NA>"]])
    filtro_status = col_f2.multiselect("Filtrar por Status:", [s for s in status_disp if s not in ["", "nan", "None", "<NA>"]])
    
    df_display = df_acao.copy()
    if filtro_contrato:
        df_display = df_display[df_display["NOME DO CONTRATO"].astype(str).isin(filtro_contrato)]
    if filtro_status:
        df_display = df_display[df_display["Status"].astype(str).isin(filtro_status)]

    st.markdown("Edite a tabela abaixo e clique em **Salvar no Google Sheets**.")
    
    df_editado = st.data_editor(
        df_display, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "NOME DO CONTRATO": st.column_config.TextColumn("NOME DO CONTRATO"),
            "What? (O que será feito)": st.column_config.TextColumn("What? (O que será feito)"),
            "Why? (Por que)": st.column_config.TextColumn("Why? (Por que)"),
            "Where? (Onde/Equipamento)": st.column_config.TextColumn("Where? (Onde/Equipamento)"),
            "Who? (Responsável)": st.column_config.TextColumn("Who? (Responsável)"),
            "How? (Como)": st.column_config.TextColumn("How? (Como)"),
            "How Much? (Custo Estimado)": st.column_config.NumberColumn("How Much? (Custo Estimado)"),
            "When? (Prazo)": st.column_config.DateColumn("When? (Prazo)", format="DD/MM/YYYY"),
            "Prazo Original": st.column_config.DateColumn("Prazo Original", format="DD/MM/YYYY", disabled=True),
            "Nova Data": st.column_config.DateColumn("Nova Data", format="DD/MM/YYYY"),
            "Status": st.column_config.SelectboxColumn("Status", options=["Concluído", "Em Andamento", "Reprogramado", "Atrasado"], required=False),
            "Motivo": st.column_config.TextColumn("Motivo"),
            "Dias de Atraso": st.column_config.TextColumn("Dias de Atraso", disabled=True),
            "Histórico": st.column_config.TextColumn("Histórico", disabled=True)
        }
    )
    
    if st.button("💾 Salvar no Google Sheets (via Apps Script)"):
        linhas_invalidas = df_editado[
            (df_editado['Status'].isin(['Reprogramado', 'Atrasado'])) & 
            ((df_editado['Motivo'].isna()) | (df_editado['Motivo'] == "") | (df_editado['Nova Data'].isna()))
        ]
        
        if not linhas_invalidas.empty:
            st.error("⚠️ **Atenção:** Ações 'Reprogramadas' ou 'Atrasadas' exigem 'Motivo' e 'Nova Data'.")
        else:
            with st.spinner("Processando Auditoria e Enviando..."):
                try:
                    hoje = datetime.now().strftime('%d/%m/%Y')
                    
                    df_editado['Histórico'] = df_editado['Histórico'].astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
                    df_editado['Dias de Atraso'] = df_editado['Dias de Atraso'].astype(str).replace({'nan': '', 'None': '', '<NA>': ''})
                    
                    for idx, row in df_editado.iterrows():
                        if pd.isnull(row['Prazo Original']) and pd.notnull(row['When? (Prazo)']):
                            df_editado.at[idx, 'Prazo Original'] = row['When? (Prazo)']
                            
                        try:
                            prazo_orig = pd.to_datetime(df_editado.at[idx, 'Prazo Original'])
                            nova_dt = pd.to_datetime(row['Nova Data'])
                            if pd.notnull(prazo_orig) and pd.notnull(nova_dt):
                                atraso = (nova_dt - prazo_orig).days
                                df_editado.at[idx, 'Dias de Atraso'] = str(atraso) if atraso > 0 else "0"
                        except Exception:
                            pass
                        
                        old_status = str(df_acao.loc[idx, 'Status']) if idx in df_acao.index else ""
                        new_status = str(row.get('Status', ''))
                        
                        if new_status != old_status and new_status in ['Reprogramado', 'Atrasado']:
                            hist = str(row.get('Histórico', ''))
                            try:
                                nova_dt_str = pd.to_datetime(row['Nova Data']).strftime('%d/%m/%Y')
                            except:
                                nova_dt_str = ""
                            novo_reg = f"[{hoje}] Mudou p/ {new_status} (Nova Data: {nova_dt_str}). Motivo: {row.get('Motivo','')}"
                            df_editado.at[idx, 'Histórico'] = (hist + " | " + novo_reg).strip(" | ")

                    df_editado['When? (Prazo)'] = df_editado['When? (Prazo)'].apply(format_date_safe)
                    df_editado['Prazo Original'] = df_editado['Prazo Original'].apply(format_date_safe)
                    df_editado['Nova Data'] = df_editado['Nova Data'].apply(format_date_safe)
                    
                    df_acao['When? (Prazo)'] = df_acao['When? (Prazo)'].apply(format_date_safe)
                    df_acao['Prazo Original'] = df_acao['Prazo Original'].apply(format_date_safe)
                    df_acao['Nova Data'] = df_acao['Nova Data'].apply(format_date_safe)

                    df_editado = df_editado.astype(str)
                    df_unfiltered = df_acao[~df_acao.index.isin(df_editado.index)]
                    df_final = pd.concat([df_unfiltered, df_editado]).sort_index()
                    
                    df_final = df_final.fillna("")
                    df_final = df_final.astype(str).replace({'nan': '', 'None': '', 'NaT': '', '<NA>': '', 'NaN': ''})
                    
                    dados_json = df_final.to_dict(orient="records")
                    resposta = requests.post(URL_APPS_SCRIPT, json=dados_json)
                    
                    if resposta.status_code == 200 and resposta.json().get("status") == "success":
                        st.success("✅ Dados salvos e auditados no Google Sheets!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Erro do servidor Google: {resposta.text}")
                except Exception as e:
                    st.error(f"⚠️ Erro crítico ao processar JSON: {e}")

    # --- EXPORTAÇÃO EXCEL E PDF ---
    st.markdown("---")
    st.markdown("### 📥 Exportar Plano de Ação")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
            df_editado.to_excel(writer, index=False, sheet_name='5W2H')
        st.download_button("📊 Baixar em Excel (.xlsx)", data=excel_data.getvalue(), file_name='plano_acao_5w2h.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    with col_d2:
        if FPDF_INSTALLED:
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Plano de Acao 5W2H", ln=True, align='C')
            
            pdf.set_font("Arial", 'B', 7)
            pdf_headers = ["Contrato", "O que", "Por que", "Onde", "When", "Prazo Orig.", "Resp.", "Como", "Custo", "Status", "Motivo", "Nova Dt.", "Atr.", "Historico"]
            col_widths = [15, 20, 20, 15, 16, 16, 15, 25, 12, 18, 25, 16, 10, 45] 
            
            max_lines = 1
            for i, col in enumerate(pdf_headers):
                lines = get_pdf_lines(pdf, remove_accents(col), col_widths[i])
                if lines > max_lines: max_lines = lines
            row_height = max_lines * 4
            
            x, y = pdf.get_x(), pdf.get_y()
            for i, col in enumerate(pdf_headers):
                pdf.set_xy(x, y)
                pdf.multi_cell(col_widths[i], 4, remove_accents(col), border=1, align='C')
                x += col_widths[i]
            pdf.set_y(y + row_height)
            
            pdf.set_font("Arial", '', 6)
            for idx, row in df_editado.iterrows():
                row_data = [str(row.get(c, "")) for c in colunas_5w2h]
                row_data = [remove_accents(item) if item not in ["nan", "NaT", "None", "<NA>"] else "" for item in row_data]
                
                max_lines = 1
                for i, text in enumerate(row_data):
                    lines = get_pdf_lines(pdf, text, col_widths[i])
                    if lines > max_lines: max_lines = lines
                row_height = max_lines * 4
                
                x, y = pdf.get_x(), pdf.get_y()
                if y + row_height > 190:
                    pdf.add_page()
                    x, y = pdf.get_x(), pdf.get_y()
                    
                for i, text in enumerate(row_data):
                    pdf.set_xy(x, y)
                    pdf.multi_cell(col_widths[i], 4, text, border=1, align='L')
                    x += col_widths[i]
                pdf.set_y(y + row_height)
            
            pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
            st.download_button("📄 Baixar em PDF (Formatado)", data=pdf_bytes, file_name='plano_acao_5w2h.pdf', mime='application/pdf')
        else:
            st.warning("Biblioteca FPDF não instalada.")

# =====================================================================
# ABA 5: CONTROLE DE COMPONENTES E LDA
# =====================================================================
with aba_lda:
    st.header("🛠️ Análise de Dados de Vida (LDA) - Componentes")
    
    if not LIFELINES_INSTALLED:
        st.error("⚠️ Biblioteca `lifelines` ausente no requirements.txt.")
    else:
        file_lda = st.file_uploader("Carregue a planilha de Controle de Componentes", type=["xlsx"], key="lda_uploader")
        
        if file_lda is not None:
            df_comp_raw = pd.read_excel(file_lda, sheet_name=0)
            df_comp_raw.columns = df_comp_raw.columns.str.replace('\n', ' ').str.replace('  ', ' ').str.strip()
            
            if 'SITUAÇÃO DO COMPONENTE' in df_comp_raw.columns and 'HORAS TRABALHADAS DO COMPONENTE' in df_comp_raw.columns:
                
                # --- NOVOS FILTROS GLOBAIS DE CONDIÇÃO E CONFIANÇA ---
                st.markdown("#### ⚙️ Configurações e Filtros Globais da Base")
                col_cfg1, col_cfg2 = st.columns(2)
                with col_cfg1:
                    st.info("A classificação de falhas procura pela palavra-chave definida abaixo na coluna de Situação.")
                    palavra_falha = st.text_input("Qual palavra indica que o componente FALHOU?", value="falhou").lower()
                with col_cfg2:
                    st.info("O Nível de Confiança ajusta as margens de erro (Limites Inferior e Superior) das análises estatísticas.")
                    confianca_input = st.number_input("Nível de Confiança (%) - Curvas Kaplan-Meier/Weibull", min_value=1.0, max_value=99.9, value=95.0, step=1.0)
                    alpha_lda = 1.0 - (confianca_input / 100.0)

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if 'MODELO' in df_comp_raw.columns:
                        modelos_disp = df_comp_raw['MODELO'].dropna().astype(str).unique().tolist()
                        modelo_alvo = st.multiselect("Filtre pelo Modelo:", modelos_disp, default=modelos_disp)
                    else:
                        modelo_alvo = []
                
                with col_f2:
                    # Busca automaticamente colunas com a palavra 'CONDIÇÃO' (ou deixa selecionar)
                    cols_cond = [c for c in df_comp_raw.columns if 'CONDIÇÃO' in c.upper() or 'CONDICAO' in c.upper()]
                    default_col = cols_cond[0] if cols_cond else df_comp_raw.columns[0]
                    
                    col_condicao = st.selectbox("Selecione a coluna de Condição (Original, Novo, Reman, etc.):", df_comp_raw.columns, index=df_comp_raw.columns.tolist().index(default_col))
                    condicoes_disp = df_comp_raw[col_condicao].dropna().astype(str).unique().tolist()
                    condicoes_sel = st.multiselect("Filtre pela Condição:", condicoes_disp, default=condicoes_disp)

                # Aplicando os filtros globais ANTES de qualquer cálculo
                df_comp = df_comp_raw.copy()
                if modelo_alvo:
                    df_comp = df_comp[df_comp['MODELO'].astype(str).isin(modelo_alvo)]
                if condicoes_sel:
                    df_comp = df_comp[df_comp[col_condicao].astype(str).isin(condicoes_sel)]
                
                df_comp['Status_LDA'] = df_comp['SITUAÇÃO DO COMPONENTE'].apply(lambda x: 1 if isinstance(x, str) and palavra_falha in x.lower() else 0)
                df_comp['Horas_LDA'] = pd.to_numeric(df_comp['HORAS TRABALHADAS DO COMPONENTE'], errors='coerce')
                
                st.markdown("---")
                
                componentes_disp = df_comp['COMPONENTE'].dropna().unique().tolist()
                comp_alvo = st.selectbox("Selecione o Componente Específico para Análise:", componentes_disp)
                
                df_alvo = df_comp[df_comp['COMPONENTE'] == comp_alvo].dropna(subset=['Horas_LDA'])
                df_alvo = df_alvo[df_alvo['Horas_LDA'] > 0]
                
                if len(df_alvo) > 0:
                    falhas_count = df_alvo['Status_LDA'].sum()
                    susp_count = len(df_alvo) - falhas_count
                    
                    st.markdown("### 📋 Tabela Resumo da Base Filtrada")
                    cols_exist = [c for c in ['MODELO', 'TAG', 'COMPONENTE', col_condicao, 'SITUAÇÃO DO COMPONENTE', 'Horas_LDA'] if c in df_alvo.columns]
                    st.dataframe(df_alvo[cols_exist].sort_values('Horas_LDA', ascending=False), use_container_width=True)
                    
                    if falhas_count > 0:
                        # Adicionando o Alpha (Confiança) no cálculo
                        wf = WeibullFitter(alpha=alpha_lda)
                        wf.fit(df_alvo['Horas_LDA'], event_observed=df_alvo['Status_LDA'])
                        st.success(f"**Weibull (IC {confianca_input}%):** $\\beta$ = {wf.rho_:.2f} | $\\eta$ = {wf.lambda_:.2f}h | **MTTF:** {wf.lambda_ * gamma(1 + (1/wf.rho_)):.2f}h")
                        
                        kmf = KaplanMeierFitter(alpha=alpha_lda)
                        kmf.fit(df_alvo['Horas_LDA'], event_observed=df_alvo['Status_LDA'])
                        t_lda = np.linspace(0.1, df_alvo['Horas_LDA'].max() * 1.2, 100)
                        
                        tab_l1, tab_l2 = st.tabs(["Confiabilidade R(t) com Intervalo de Confiança", "Taxa de Falha h(t)"])
                        with tab_l1:
                            fig_l1 = go.Figure()
                            
                            # Renderização do preenchimento da área do Intervalo de Confiança (Kaplan-Meier)
                            ci_lower = kmf.confidence_interval_.iloc[:, 0]
                            ci_upper = kmf.confidence_interval_.iloc[:, 1]
                            
                            fig_l1.add_trace(go.Scatter(x=ci_lower.index, y=ci_lower, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                            fig_l1.add_trace(go.Scatter(x=ci_upper.index, y=ci_upper, mode='lines', fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=f'IC {confianca_input}% (KM)'))
                            
                            # Linhas centrais
                            fig_l1.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'], mode='lines', line=dict(shape='hv', color='blue'), name='Kaplan-Meier (Central)'))
                            fig_l1.add_trace(go.Scatter(x=t_lda, y=wf.survival_function_at_times(t_lda), mode='lines', line=dict(dash='dash', color='red'), name='Weibull'))
                            fig_l1.update_layout(xaxis_title="Tempo (Horas)", yaxis_title="Probabilidade de Sobrevivência", hovermode="x unified")
                            st.plotly_chart(fig_l1, use_container_width=True)
                            
                        with tab_l2:
                            st.plotly_chart(go.Figure(go.Scatter(x=t_lda, y=wf.hazard_at_times(t_lda), mode='lines', line=dict(color='purple'))), use_container_width=True)

                st.markdown("---")
                st.markdown("### 🔥 Mapa de Calor: Vida Útil Restante da Frota")
                
                # Recalcula usando os filtros globais (já aplicados) e enviando o Alpha atualizado
                component_metrics = calcular_metricas_componentes(df_comp, alpha=alpha_lda)
                
                df_ativos = df_comp[(df_comp['Status_LDA'] == 0) & (df_comp['Horas_LDA'] > 0)].copy()
                df_ativos['MTTF'] = df_ativos['COMPONENTE'].map(lambda c: component_metrics.get(c, {}).get('MTTF', np.nan))
                df_ativos['Prob_MTTF'] = df_ativos['COMPONENTE'].map(lambda c: component_metrics.get(c, {}).get('Prob_MTTF', np.nan))
                df_ativos = df_ativos.dropna(subset=['MTTF'])

                if not df_ativos.empty:
                    df_ativos['Vida_Consumida_%'] = (df_ativos['Horas_LDA'] / df_ativos['MTTF']) * 100
                    if 'TAG' in df_ativos.columns:
                        df_ativos['EQUIP'] = df_ativos['TAG'].astype(str).apply(lambda x: x.split(' ')[0])
                    else:
                        df_ativos['EQUIP'] = 'Geral'
                        
                    if 'DATA' in df_ativos.columns:
                        df_ativos['Hover_Text'] = df_ativos.apply(
                            lambda row: f"<b>Equipamento:</b> {row['EQUIP']}<br>"
                                        f"<b>Componente:</b> {row['COMPONENTE']}<br>"
                                        f"<b>Condição Base:</b> {row.get(col_condicao, '')}<br>"
                                        f"<b>Vida Consumida:</b> {row['Vida_Consumida_%']:.1f}%<br>"
                                        f"<b>Horímetro Atual:</b> {row['Horas_LDA']}h<br>"
                                        f"<b>MTTF Estimado:</b> {row['MTTF']:.1f}h<br>"
                                        f"<b>Data de Leitura:</b> {format_date_safe(row.get('DATA', 'N/A'))}", axis=1
                        )
                    else:
                        df_ativos['Hover_Text'] = df_ativos.apply(
                            lambda row: f"<b>Equipamento:</b> {row['EQUIP']}<br>"
                                        f"<b>Componente:</b> {row['COMPONENTE']}<br>"
                                        f"<b>Condição Base:</b> {row.get(col_condicao, '')}<br>"
                                        f"<b>Vida Consumida:</b> {row['Vida_Consumida_%']:.1f}%<br>"
                                        f"<b>Horímetro Atual:</b> {row['Horas_LDA']}h<br>"
                                        f"<b>MTTF Estimado:</b> {row['MTTF']:.1f}h", axis=1
                        )
                    
                    col_h1, col_h2, col_h3 = st.columns(3)
                    with col_h1:
                        max_vida = float(df_ativos['Vida_Consumida_%'].max()) if not df_ativos.empty else 100.0
                        faixa_vida = st.slider("Filtro Visual - % de Vida Consumida:", min_value=0.0, max_value=max(200.0, max_vida), value=(0.0, max(100.0, max_vida)))
                    
                    with col_h2:
                        comps_heatmap = df_ativos['COMPONENTE'].unique().tolist()
                        selecao_comps = st.multiselect("Esconder Componentes Específicos:", comps_heatmap, default=comps_heatmap)
                        
                    with col_h3:
                        situacoes_disp = df_ativos['SITUAÇÃO DO COMPONENTE'].dropna().unique().tolist()
                        selecao_situacao = st.multiselect("Esconder Situações (ex: Preventivas):", situacoes_disp, default=situacoes_disp)

                    df_heat_filtered = df_ativos[
                        (df_ativos['Vida_Consumida_%'] >= faixa_vida[0]) & 
                        (df_ativos['Vida_Consumida_%'] <= faixa_vida[1]) & 
                        (df_ativos['COMPONENTE'].isin(selecao_comps)) &
                        (df_ativos['SITUAÇÃO DO COMPONENTE'].isin(selecao_situacao))
                    ]

                    if not df_heat_filtered.empty:
                        heatmap_data = df_heat_filtered.pivot_table(index='EQUIP', columns='COMPONENTE', values='Vida_Consumida_%', aggfunc='mean').fillna(0)
                        hover_data = df_heat_filtered.pivot_table(index='EQUIP', columns='COMPONENTE', values='Hover_Text', aggfunc='first').fillna("")
                        
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=heatmap_data.values,
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            text=heatmap_data.values.round(1),
                            texttemplate="%{text}",
                            customdata=hover_data.values,
                            hovertemplate="%{customdata}<extra></extra>",
                            colorscale="RdYlGn_r"
                        ))
                        fig_heat.update_layout(title="Porcentagem (%) do MTTF Consumida", xaxis_title="COMPONENTE", yaxis_title="EQUIP")
                        st.plotly_chart(fig_heat, use_container_width=True)
                        
                        df_mttf_display = pd.DataFrame.from_dict(component_metrics, orient='index').reset_index()
                        df_mttf_display.columns = ['Componente', 'MTTF Calculado (Horas)', 'Probabilidade Falha no MTTF (%)']
                        df_mttf_display = df_mttf_display[df_mttf_display['Componente'].isin(selecao_comps)].sort_values('MTTF Calculado (Horas)')
                        
                        df_mttf_display['MTTF Calculado (Horas)'] = df_mttf_display['MTTF Calculado (Horas)'].round(2)
                        df_mttf_display['Probabilidade Falha no MTTF (%)'] = df_mttf_display['Probabilidade Falha no MTTF (%)'].round(2)
                        
                        st.dataframe(df_mttf_display, use_container_width=True)
                    else:
                        st.warning("Nenhum componente ativo encontrado para os filtros selecionados.")
            else:
                st.error("Colunas obrigatórias ausentes (SITUAÇÃO DO COMPONENTE, HORAS TRABALHADAS DO COMPONENTE).")
