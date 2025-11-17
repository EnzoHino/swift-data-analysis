# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk
import re
from datetime import datetime

# Se necessário, descomente para baixar stopwords uma vez
# nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('portuguese'))

CUSTOM_STOPWORDS = {
    "ta", "tá", "tah", "tava", "so", "mt", "pra", "pro", "ser", "será",
    "eh", "é", "nao", "não", "n", "q", "kkk", "kk", "tipo", "ai", "aq",
    "ne", "né", "deu", "dei", "tive", "tava", "ate", "até", "ja", "já"
}
 
ALL_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# -----------------------
# Config Streamlit / tema laranja
# -----------------------
st.set_page_config(layout="wide", page_title="Dash NPS - Swift", page_icon="🟠")
st.markdown(
    """
    <style>

    /* ===============================
       BACKGROUND GERAL (MODO DARK)
       =============================== */
    .stApp {
        background-color: #111111;
        color: #EEEEEE !important;
    }

    /* ===============================
       SIDEBAR
       =============================== */
    section[data-testid="stSidebar"] {
        background-color: #181818 !important;
        border-right: 1px solid #333 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #E4E4E4 !important;
    }

    /* Títulos da sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label {
        color: #FFA045 !important;
        font-weight: 600;
    }

    /* ===============================
       TÍTULOS E TEXTOS DO MAIN
       =============================== */
    h1, h2, h3, h4 {
        color: #FFA045 !important;
    }

    p, span, div, label {
        color: #DDDDDD !important;
    }

    /* ===============================
       TABS
       =============================== */
    .stTabs [role="tab"] {
        background: #222 !important;
        color: #EEE !important;
        border: 1px solid #333 !important;
        padding: 8px 20px;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: #FFA045 !important;
        color: black !important;
        font-weight: 700;
        border-bottom: 2px solid black !important;
    }

    /* ===============================
       BOTÕES
       =============================== */
    .stButton>button {
        background-color: #FFA045 !important;
        color: #000 !important;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 8px 15px;
    }

    .stButton>button:hover {
        background-color: #ffbb6a !important;
        color: #000 !important;
    }

    /* ===============================
       SELECTBOXES / INPUTS
       =============================== */
    .stSelectbox div, .stMultiSelect div, .stTextInput div {
        color: #EEE !important;
    }

    /* Inputs */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        background: #222 !important;
    }

    /* ===============================
       MÉTRICAS
       =============================== */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }

    div[data-testid="metric-container"] * {
        color: #FFA045 !important;
    }

    /* ===============================
       DATAFRAMES
       =============================== */
    .dataframe {
        background-color: #222 !important;
        color: #EEE !important;
    }

    /* Plotly background fix */
    .js-plotly-plot .plotly, 
    .js-plotly-plot .plot-container {
        background-color: #111111 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Painel de Análise de Sentimentos e Bigramas — Swift")

# -----------------------
# Data load
# -----------------------
@st.cache_data
def load_data(path="Comentarios.csv"):
    df = pd.read_csv(path, parse_dates=['Data Avaliação'], dayfirst=True, infer_datetime_format=True)
    # ensure expected columns exist
    expected_cols = ['Ano','Centro','Mes','Semana','Data Avaliação','NPS','Categoria','final_text']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")
    
    df['Data Avaliação'] = pd.to_datetime(
    df['Data Avaliação'],
    errors='coerce',   # valores inválidos viram NaT
    dayfirst=True)
    # normalize text
    df['final_text'] = df['final_text'].astype(str)
    df['text_clean'] = df['final_text'].apply(lambda x: normalize_text(x))
    # ensure categorical types
    df['NPS'] = df['NPS'].astype(str)
    df['Categoria'] = df['Categoria'].astype(str)
    df['Centro'] = df['Centro'].astype(str)
    # add year-month for time series
    df['year_month'] = df['Data Avaliação'].dt.to_period('M').astype(str)
    return df

# Normalização simples
def normalize_text(text):
    text = text.lower()
 
    # Remove links
    text = re.sub(r'http\S+', ' ', text)
 
    # Remove números
    text = re.sub(r'\d+', ' ', text)
 
    # Remove tudo que não for letra
    text = re.sub(r'[^a-záéíóúãõâêôç\s]', ' ', text)
 
    # Remove tokens de 1 letra
    text = re.sub(r'\b[a-z]\b', ' ', text)
 
    # Normaliza espaços
    text = re.sub(r'\s+', ' ', text).strip()
    return text

 

# Bigram extraction
def get_top_bigrams(corpus, n=30, min_df=2, stopwords=ALL_STOPWORDS):
    vect = CountVectorizer(
        ngram_range=(2,2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=1
    )
 
    X = vect.fit_transform(corpus)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = vect.get_feature_names_out()
 
    filtered = []
    for token, count in zip(vocab, counts):
        w1, w2 = token.split()
 
        # remover tokens de 1 letra
        if len(w1) == 1 or len(w2) == 1:
            continue
 
        # remover tokens numéricos
        if w1.isdigit() or w2.isdigit():
            continue
 
        # remover palavras que contêm números
        if any(ch.isdigit() for ch in w1) or any(ch.isdigit() for ch in w2):
            continue
 
        # remover bigramas com stopwords customizadas
        if w1 in stopwords or w2 in stopwords:
            continue
 
        filtered.append((token, int(count)))
 
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]

# Generic helper to present bigram table
def bigram_df(bigrams):
    df = pd.DataFrame(bigrams, columns=['Bigrama','Contagem'])
    df.index = df.index + 1
    return df

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filtros")

try:
    df = load_data("./Comentarios.csv")
except Exception as e:
    st.sidebar.error(e)
    st.sidebar.error("Arquivo 'Comentarios.csv' não encontrado. Faça upload do arquivo ou salve seu CSV como 'Comentarios.csv' na pasta do app.")
    st.stop()

centros = ['Todos'] + sorted(df['Centro'].unique().tolist())
centro_sel = st.sidebar.selectbox("Loja (Centro)", centros, index=0)
anos = ['Todos'] + sorted(df['Ano'].dropna().unique().astype(int).astype(str).tolist())
ano_sel = st.sidebar.selectbox("Ano", anos, index=0)
categorias = ['Todos'] + sorted(df['Categoria'].unique().tolist())
cat_sel = st.sidebar.selectbox("Categoria", categorias, index=0)
n_top = st.sidebar.slider("Quantos top bigramas exibir", min_value=5, max_value=20, value=20, step=1)

# Data filtering
df_f = df.copy()
if centro_sel != 'Todos':
    df_f = df_f[df_f['Centro']==centro_sel]
if ano_sel != 'Todos':
    df_f = df_f[df_f['Ano']==int(ano_sel)]
if cat_sel != 'Todos':
    df_f = df_f[df_f['Categoria']==cat_sel]

# -----------------------
# Layout: Tabs
# -----------------------
tabs = st.tabs(["Visão Geral", "Bigramas", "Categorias & NPS", "Séries Temporais", "Export & Ações"])

# ---- Tab 0: Visão Geral ----
with tabs[0]:
    st.header("Visão Geral dos Dados")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avaliações", f"{len(df_f):,}")
    col2.metric("Promotores", f"{(df_f['NPS']=='Positivo').sum():,}")
    col3.metric("Neutros", f"{(df_f['NPS']=='Neutro').sum():,}")
    col4.metric("Detratores", f"{(df_f['NPS']=='Detrator').sum():,}")

    st.markdown("**Distribuição de Sentimentos (NPS)**")
    fig_pie = px.pie(df_f, names='NPS', title="Proporção por NPS", hole=0.35,
                     color_discrete_sequence=px.colors.sequential.Oranges)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Top categorias por volume**")
    topcat = df_f['Categoria'].value_counts().reset_index()
    topcat.columns = ['Categoria','count']
    fig_bar = px.bar(topcat.head(20), x='count', y='Categoria', orientation='h',
                     title="Categorias mais comentadas", height=500,
                     color_discrete_sequence=px.colors.sequential.Oranges)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---- Tab 1: Bigramas ----
with tabs[1]:
    st.header("Análise de Bigramas")
    subtab = st.tabs(["Por Polaridade (NPS)", "Por Categoria", "Por Sentimento", "Negativo por Categoria"])

    # 1. Bigrams por polaridade (promotor, neutro, detrator)
    with subtab[0]:
        st.subheader("1) Bigramas mais frequentes por polaridade (NPS)")
        cols = st.columns(3)
        for i, nps in enumerate(['Positivo','Neutro','Detrator']):
            with cols[i]:
                df_n = df_f[df_f['NPS']==nps]
                bigs = get_top_bigrams(df_n['text_clean'].values.astype('U'), n=n_top)
                st.markdown(f"**{nps} — total avaliações:** {len(df_n):,}")
                st.table(bigram_df(bigs).head(20))

    # 2. Bigrams por categoria
    with subtab[1]:
        st.subheader("2) Bigramas mais recorrentes em cada Categoria")
        cat_sel_tab = st.selectbox("Escolha categoria (aba Bigramas)", ['Todas'] + sorted(df_f['Categoria'].unique().tolist()))
        if cat_sel_tab == 'Todas':
            # show grid of top categories
            top_cats = df_f['Categoria'].value_counts().head(6).index.tolist()
        else:
            top_cats = [cat_sel_tab]
        for c in top_cats:
            st.markdown(f"**Categoria: {c}** — quantidade: {df_f[df_f['Categoria']==c].shape[0]}")
            bigs = get_top_bigrams(df_f[df_f['Categoria']==c]['text_clean'].astype('U'), n=n_top) if df_f[df_f['Categoria']==c].shape[0]>0 else []
            st.table(bigram_df(bigs).head(20))

    # 3. Bigrams por sentimento (sentimento derivado do texto: Positivo/Negativo)
    # -> We'll assume NPS is the sentiment, but we'll also allow an automatic rule: NPS==Positivo => positivo; Detrator => negativo; Neutro => neutro.
    with subtab[2]:
        st.subheader("3) Bigramas por sentimento (usando NPS como proxy)")
        sent_map = {'Positivo':'Positivo','Neutro':'Neutro','Detrator':'Negativo'}
        for label, group in [('Positivo', df_f[df_f['NPS']=='Positivo']),
                             ('Neutro', df_f[df_f['NPS']=='Neutro']),
                             ('Negativo (Detrator)', df_f[df_f['NPS']=='Detrator'])]:
            st.markdown(f"**{label}** — {len(group):,} avaliações")
            bigs = get_top_bigrams(group['text_clean'].astype('U'), n=n_top)
            st.table(bigram_df(bigs).head(20))

    # 4. Bigramas mais recorrentes no sentimento negativo em cada categoria
    with subtab[3]:
        st.subheader("4) Bigramas mais recorrentes no sentimento NEGATIVO em cada categoria")
        cat_sel_neg = st.selectbox("Escolha (negativos por categoria)", ['Todas'] + sorted(df_f['Categoria'].unique().tolist()), key='neg_cat')
        cats_to_show = [cat_sel_neg] if cat_sel_neg!='Todas' else df_f['Categoria'].value_counts().head(8).index.tolist()
        for c in cats_to_show:
            grp = df_f[(df_f['Categoria']==c) & (df_f['NPS']=='Detrator')]
            st.markdown(f"**{c}** — negativas: {len(grp):,}")
            if len(grp)>0:
                bigs = get_top_bigrams(grp['text_clean'].astype('U'), n=n_top)
                st.table(bigram_df(bigs).head(20))
            else:
                st.write("Sem comentários negativos nessa categoria (no filtro atual).")

# ---- Tab 2: Categorias & NPS ----
with tabs[2]:
    st.header("Categorias: concentração de Detratores / Promotores e sentimento")
    st.markdown("**6 / 7 / 8 / 9 — categorização por concentração**")
    # compute counts by category and NPS
    cat_nps = df_f.groupby(['Categoria','NPS']).size().reset_index(name='count')
    pivot = cat_nps.pivot(index='Categoria', columns='NPS', values='count').fillna(0)
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('Total', ascending=False)
    pivot_display = pivot.reset_index()
    pivot_display.index = pivot_display.index + 1

    st.dataframe(pivot_display.head(50))

    # Which categories concentrate most detratores?
    st.subheader("Categorias com mais Detratores")
    if 'Detrator' in pivot.columns:
        top_detr = pivot.sort_values('Detrator', ascending=False).reset_index().head(10)
        fig_d = px.bar(top_detr, x='Detrator', y='Categoria', orientation='h', title="Top categorias por Detratores",
                       color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.write("Sem registros de 'Detrator' no conjunto filtrado.")

    # Which categories concentrate mais promotores?
    st.subheader("Categorias com mais Promotores")
    if 'Positivo' in pivot.columns:
        top_prom = pivot.sort_values('Positivo', ascending=False).reset_index().head(10)
        fig_p = px.bar(top_prom, x='Positivo', y='Categoria', orientation='h', title="Top categorias por Promotores",
                       color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.write("Sem registros de 'Positivo' no conjunto filtrado.")

    # categories with most negative comments (here same as detratores)
    st.subheader("Categorias com mais comentários NEGATIVOS (Detratores)")
    if 'Detrator' in pivot.columns:
        fig_neg = px.bar(pivot.reset_index().head(20), x='Detrator', y='Categoria', orientation='h',
                         title="Distribuição de Comentários Negativos por Categoria",
                         color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig_neg, use_container_width=True)

    # categories with most positive comments
    st.subheader("Categorias com mais comentários POSITIVOS")
    if 'Positivo' in pivot.columns:
        fig_pos = px.bar(pivot.reset_index().head(20), x='Positivo', y='Categoria', orientation='h',
                         title="Distribuição de Comentários Positivos por Categoria",
                         color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig_pos, use_container_width=True)

# ---- Tab 3: Séries Temporais ----
with tabs[3]:
    st.header("Evolução Temporal do Sentimento")
    st.markdown("10) Sentimento (positivo/negativo) ao longo do tempo")

    # formatar ano-mes
    df_f['year_month_fmt'] = pd.to_datetime(df_f['year_month']).dt.strftime('%m/%Y')

    ts = df_f.groupby(['year_month_fmt','NPS']).size().reset_index(name='count')

    ts_pivot = ts.pivot(index='year_month_fmt', columns='NPS', values='count').fillna(0)
    ts_pivot = ts_pivot.sort_index()

    fig_ts = px.area(ts_pivot.reset_index(), 
                     x='year_month_fmt', 
                     y=ts_pivot.columns.tolist(),
                     title="Volume de avaliações por sentimento ao longo do tempo")

    fig_ts.update_layout(legend_title_text="Classificação")
    fig_ts.update_yaxes(title_text="Contagem")
    fig_ts.update_xaxes(title_text="Mês/Ano")
    
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("11) Sentimento dos Detratores e Promotores ao longo do tempo")

    ts_dp = df_f[df_f['NPS'].isin(['Detrator','Positivo'])] \
                .groupby(['year_month_fmt','NPS']) \
                .size().reset_index(name='count')

    ts_dp_p = ts_dp.pivot(index='year_month_fmt', columns='NPS', values='count') \
                   .fillna(0).sort_index()

    fig_dp = px.line(ts_dp_p.reset_index(), 
                     x='year_month_fmt', 
                     y=ts_dp_p.columns.tolist(),
                     title="Detratores vs Promotores ao longo do tempo")

    fig_dp.update_layout(legend_title_text="Classificação")
    fig_dp.update_yaxes(title_text="Contagem")
    fig_dp.update_xaxes(title_text="Mês/Ano")



    st.plotly_chart(fig_dp, use_container_width=True)


# ---- Tab 4: Export & Ações ----
with tabs[4]:
    st.header("Exportar resultados e recomendações (Ações sugeridas)")
    st.markdown("Você pode exportar as tabelas de bigramas e summaries das categorias para CSV.")

    if st.button("Exportar top bigramas por NPS (CSV)"):
        # build csv
        out_rows = []
        for nps in ['Positivo','Neutro','Detrator']:
            gf = df_f[df_f['NPS']==nps]
            bigs = get_top_bigrams(gf['text_clean'].astype('U'), n=n_top)
            for big, cnt in bigs:
                out_rows.append({'NPS':nps,'bigram':big,'count':cnt})
        out_df = pd.DataFrame(out_rows)
        out_name = "top_bigrams_by_nps.csv"
        out_df.to_csv(out_name, index=False)
        st.success(f"Arquivo salvo: {out_name}")

    if st.button("Exportar pivot categorias x NPS (CSV)"):
        pivot_display.to_csv("pivot_categoria_nps.csv", index=False)
        st.success("Arquivo salvo: pivot_categoria_nps.csv")

    st.markdown("### 12) Ações sugeridas (resumo automático baseado nos achados)")
    st.write("As ações abaixo são geradas a partir das análises (bigramas frequentes e categorias com mais detratores). Ajuste conforme contexto da loja:")

    # Simple auto recommender based on top negative bigrams and categories
    # find top negative bigrams global
    neg_bigs = get_top_bigrams(df_f[df_f['NPS']=='Detrator']['text_clean'].astype('U'), n=30)
    top_neg_bigrams = [b for b,c in neg_bigs[:10]]
    st.markdown("**Sugestões automáticas**:")
    st.markdown(f"- **Problemas identificados com frequência nos comentários negativos (exemplos de bigramas):** {', '.join(top_neg_bigrams[:8])}")
    st.markdown("- **Ações rápidas recomendadas:**")
    st.write("""
    1. Revisar estoque das categorias com maior volume de detratores — executar inventário e priorizar reposição dos itens citados.
    2. Treinamento rápido ao time de atendimento para as categorias com mais reclamações (script + resolução imediata).
    3. Monitoramento semanal (KPI) de detratores por categoria; metas de redução mês a mês.
    4. Implementar rotinas de verificação da sinalização de falta de produto no PDV e no sistema.
    5. Responder comentários negativos com um fluxo padrão (pedido de desculpas + solução + follow-up).
    6. Se bigramas negativos referirem preço/valor, revisar política de promoções/etiquetagem.
    7. Para bigramas positivos, amplificar via comunicação (ex.: promoções nas categorias muito elogiadas).
    """)

    st.info("Essas ações devem ser priorizadas conforme volume de comentários/impacto operacional. Combine com dados operacionais (venda por SKU, prazo de entrega, etc.) para priorizar.")

