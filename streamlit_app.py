import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns

################## CARREGAR FICHEIROS ################################
base = os.path.dirname(os.path.abspath(__file__))

# Caminho completo para o ficheiro
file = os.path.join(base, "dados_streamlit", "termos_titulos.csv")

# Verificar se o ficheiro existe antes de abrir
if os.path.exists(file):
    # Ler o ficheiro CSV com pandas
    termos_titulos = pd.read_csv(file)
    print("Ficheiro carregado com sucesso!")
else:
    print(f"Ficheiro não encontrado: {file}")

base = os.path.dirname(os.path.abspath(__file__))

# Caminho completo para o ficheiro
file = os.path.join(base, "dados_streamlit", "df_combined.csv")

# Verificar se o ficheiro existe antes de abrir
if os.path.exists(file):
    # Ler o ficheiro CSV com pandas
    df_all = pd.read_csv(file)
    print("Ficheiro carregado com sucesso!")
else:
    print(f"Ficheiro não encontrado: {file}")

base = os.path.dirname(os.path.abspath(__file__))

# Caminho completo para o ficheiro
file = os.path.join(base, "dados_streamlit", "df_authors.csv")

# Verificar se o ficheiro existe antes de abrir
if os.path.exists(file):
    # Ler o ficheiro CSV com pandas
    df_authors = pd.read_csv(file)
    print("Ficheiro carregado com sucesso!")
else:
    print(f"Ficheiro não encontrado: {file}")

file = os.path.join(base, "dados_streamlit", "df_combined.csv")
# Verificar se o ficheiro existe antes de abrir
if os.path.exists(file):
    # Ler o ficheiro CSV com pandas
    df_combined = pd.read_csv(file)
    print("Ficheiro carregado com sucesso!")
else:
    print(f"Ficheiro não encontrado: {file}")

print(df_combined.head())


top_10_techniques_by_cluster = pd.read_csv(r'/workspaces/Dashboard_ICD/dados_streamlit/top_10_techniques_by_cluster.csv')

################### PREPARAÇÃO DOS GRÁFICOS ##########################

# URL do VOSviewer
url = "https://tinyurl.com/233s5fwm"

# Definir as cores customizadas
custom_colors = [
    "#003f5b", "#2b4b7d", "#5f5195", "#98509d", 
    "#cc4c91", "#f25375", "#ff6f4e", "#ff9913"]

# Criar o gráfico de barras interativo para visualizar os 10 primeiros termos
fig = px.bar(
    termos_titulos.head(10),
    x='Count',
    y='Term',
    orientation='h',
    labels={'Count': 'Frequência', 'Term': 'Termo'},
    color='Count',
    color_continuous_scale=custom_colors
)

# Ajustar o layout do gráfico
fig.update_layout(
    xaxis_title='Frequência',
    yaxis_title='Termos',
    yaxis=dict(autorange='reversed'),  # Inverter a ordem dos termos no eixo y
    height=500,  # Aumentar a altura do gráfico
    margin=dict(l=30, r=30, t=40, b=40)  # Ajustar margens para dar mais espaço
)

# Layout do Streamlit
st.set_page_config(
    page_title="Dashboard - Data Science e Big Data no Planeamento Urbano",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

df_all['count'] = 1  # Inicializar com 1 para representar cada artigo
df_all = df_all.groupby(['ano', 'affiliation-country'], as_index=False).agg({'count': 'sum'})

# Encontrar o valor máximo de 'count' em qualquer país ao longo de todos os anos
max_count = df_all['count'].max()

# Criar o mapa interativo
fig1 = px.choropleth(
    df_all,
    locations="affiliation-country",  # Nome do país
    locationmode="country names",  # Usando nomes de países
    color="count",  # Usando a coluna 'count' para a cor
    color_continuous_scale=custom_colors,
    hover_name="affiliation-country",  # Para exibir o nome do país
    animation_frame="ano",  # Para animar o mapa ao longo dos anos
    range_color=[0, max_count]  # Define a escala de cor de 0 até o valor máximo fixo
)

# Ajustar o layout do mapa com fundo preto
fig1.update_layout(
    geo=dict(
        showframe=False, 
        showcoastlines=True, 
        projection_type='natural earth',
        bgcolor="black"  # Define o fundo do mapa como preto
    ),
    coloraxis_colorbar=dict(title="Número de Artigos"),
    paper_bgcolor="black",  # Fundo da área de papel também preto
    plot_bgcolor="black"    # Fundo do gráfico também preto
)


# Definir as colunas obrigatórias
# Definição das colunas e dados (ajustar conforme necessário)
required_columns = ['author', 'n_artigos_pub', 'affiliation', 'country']
if not all(column in df_authors.columns for column in required_columns):
    raise ValueError(f"O DataFrame deve conter as colunas: {required_columns}")

# Criar uma coluna com as siglas das instituições
df_authors['institution_labels_simp'] = df_authors['affiliation'].apply(lambda x: ''.join([word[0].upper() for word in x.split()]))

# Selecionar os 10 autores com mais publicações
top_authors_df = df_authors.nlargest(10, 'n_artigos_pub')

# Preparar os dados para o diagrama de Sankey
authors = top_authors_df['author'].tolist()
affiliations = top_authors_df['affiliation'].tolist()
countries = top_authors_df['country'].dropna().unique().tolist()
institution_labels_simp = [df_authors[df_authors['affiliation'] == inst]['institution_labels_simp'].values[0] for inst in affiliations]

# Criar labels para países, instituições (simplificadas) e top 10 autores
labels = countries + institution_labels_simp + authors

# Criar dicionários para mapear os índices
country_indices = {country: i for i, country in enumerate(countries)}
affiliation_indices = {affiliation: i + len(countries) for i, affiliation in enumerate(institution_labels_simp)}
author_indices = {author: i + len(countries) + len(institution_labels_simp) for i, author in enumerate(authors)}

# Criar listas para as fontes e destinos
sources = []
targets = []
values = []
link_colors = []

# Definir cores para os países
custom_colors = [
    "#5f5195", "#98509d", 
    "#cc4c91", "#f25375",
    "#ff6f4e", "#ff9913"
]
country_colors = custom_colors[:len(countries)]

# Adicionar relações país -> instituição
for i, row in top_authors_df.iterrows():
    sources.append(country_indices[row['country']])
    targets.append(affiliation_indices[institution_labels_simp[affiliations.index(row['affiliation'])]])
    values.append(row['n_artigos_pub'])
    link_colors.append(country_colors[countries.index(row['country'])])

# Adicionar relações instituição -> autor
for i, row in top_authors_df.iterrows():
    sources.append(affiliation_indices[institution_labels_simp[affiliations.index(row['affiliation'])]])
    targets.append(author_indices[row['author']])
    values.append(row['n_artigos_pub'])
    # Criar uma cor mais clara para o autor
    base_color = pc.hex_to_rgb(country_colors[countries.index(row['country'])])
    lighter_color = pc.find_intermediate_color(base_color, (255, 255, 255), 0.5)
    lighter_color = f"rgb{lighter_color}"
    link_colors.append(lighter_color)

# Criar customdata para incluir o nome completo e o número de artigos
customdata = []
for label in labels:
    if label in authors:
        count = top_authors_df[top_authors_df['author'] == label]['n_artigos_pub'].values[0]
        customdata.append(f"{label}<br>Number of articles: {count}")
    elif label in institution_labels_simp:
        full_name = affiliations[institution_labels_simp.index(label)]
        count = top_authors_df[top_authors_df['affiliation'] == full_name]['n_artigos_pub'].sum()
        customdata.append(f"{full_name}<br>Number of articles: {count}")
    else:
        count = top_authors_df[top_authors_df['country'] == label]['n_artigos_pub'].sum()
        customdata.append(f"{label}<br>Number of articles: {count}")

# Criar o diagrama de Sankey
fig2 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,  # Espessura dos nós ajustada
        line=dict(color="white", width=0),  # Remover o contorno dos nós (sem outline)
        label=labels,
        customdata=customdata,  # Texto completo para o pop-up
        hovertemplate='%{customdata}<extra></extra>',  # Formato do pop-up
        color="#003f5c",  # Cor dos nós ajustada para cinza
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

# Ajuste do layout para tornar o gráfico mais clean e bonito
fig2.update_layout(
    hovermode='x',
    font=dict(size=14, color='Black', family="Arial"),  # Aumentar o tamanho da fonte
    plot_bgcolor='black',  # Fundo preto
    paper_bgcolor='black',  # Fundo da página também preto
    width=700,  # Largura do gráfico mais ajustada
    height=500,  # Altura ajustada para um aspecto mais equilibrado
    showlegend=False  # Remover a legenda para não sobrecarregar o gráfico
)

# wordcloud

#wordcloud
custom_colors = [
    "#5f5195", "#98509d", 
    "#cc4c91", "#f25375",
    "#ff6f4e", "#ff9913"
]
df_combined['processedAbstract'] = df_combined['processedAbstract'].str.replace(r"[']", "", regex=True)
df_combined['processedAbstract'] = df_combined['processedAbstract'].str.replace(r"[,]", "", regex=True)
df_combined['processedAbstract'] = df_combined['processedAbstract'].str.replace(r"[.]", "", regex=True)

# Função que mapeia a cor das palavras
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return custom_colors[hash(word) % len(custom_colors)]

# Gerar a wordcloud
plt.figure(figsize=(16,13))

wordcloud = WordCloud(
    background_color='black',
    max_words=10000,
    width=800,
    height=400,
    color_func=color_func  # Definir a função de cores personalizada
).generate(" ".join(df_combined['processedAbstract']))

# Criar o gráfico no matplotlib
fig5, ax = plt.subplots(figsize=(10, 6), facecolor='black')  # Define a moldura preta
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Remove os eixos
ax.set_facecolor('black')  # Define o fundo do gráfico como preto

num_topics = 4

# Contagem de artigos por país
country_counts = df_combined['affiliation-country'].value_counts()

# Selecionando os 10 países com mais artigos
top_10_countries = country_counts.head(10).index

# Criando um DataFrame para contar os tópicos por país
topic_counts = pd.DataFrame(columns=[f'Tópico {i}' for i in range(num_topics)], index=top_10_countries)

# Inicializando as contagens dos tópicos com 0
for country in top_10_countries:
    topic_counts.loc[country] = np.zeros(num_topics)

# Contagem dos tópicos por país
for i, row in df_combined.iterrows():
    country = row['affiliation-country']
    if country in top_10_countries:
        topic_dist = row['topic_distribution']  # Supondo que 'topic_distribution' pode ser uma string ou uma lista de tuplas
        
        # Verificando se 'topic_dist' é uma string (caso seja uma representação de lista)
        if isinstance(topic_dist, str):
            try:
                topic_dist = eval(topic_dist)  # Converte a string para uma lista de tuplas
            except Exception as e:
                print(f"Erro ao tentar converter 'topic_distribution' para lista em {country}. Valor: {topic_dist}, Erro: {e}")
                continue
        
        # Verificando se 'topic_dist' é uma lista de tuplas
        if isinstance(topic_dist, list) and all(isinstance(item, tuple) and len(item) == 2 for item in topic_dist):
            for topic, prob in topic_dist:
                # Asegurando que 'topic' é um índice válido e 'prob' é numérico
                if isinstance(topic, int) and 0 <= topic < num_topics and isinstance(prob, (int, float)):
                    topic_counts.loc[country, f'Tópico {topic}'] += prob
                else:
                    print(f"Distribuição de tópico inválida encontrada em {country}. Tópico: {topic}, Probabilidade: {prob}")
        else:
            print(f"'topic_distribution' não é uma lista de tuplas válida em {country}. Valor encontrado: {topic_dist}")

# Garantindo que todos os valores de 'topic_counts' são numéricos e substituindo NaNs por 0
topic_counts = topic_counts.apply(pd.to_numeric, errors='coerce').fillna(0)

# Normalizando os valores para percentagem
topic_counts_percentage = topic_counts.div(topic_counts.sum(axis=1), axis=0) * 100

# Contagem dos tópicos por ano
topic_counts_by_year = pd.DataFrame(0, index=df_combined['ano'].unique(), columns=[f'Topic {i}' for i in range(num_topics)])

# Preenchendo a tabela com a distribuição dos tópicos por ano
for idx, row in df_combined.iterrows():
    year = row['ano']
    topic_dist = row['topic_distribution']  # Distribuição de tópicos (lista de probabilidades)
    if isinstance(topic_dist, str):
        try:
            topic_dist = eval(topic_dist)  # Converte a string para uma lista de tuplas
        except Exception as e:
            print(f"Erro ao tentar converter 'topic_distribution' para lista no ano {year}. Valor: {topic_dist}, Erro: {e}")
            continue
    # Verificando se 'topic_dist' é uma lista de tuplas
    if isinstance(topic_dist, list) and all(isinstance(item, tuple) and len(item) == 2 for item in topic_dist):
        for topic, prob in topic_dist:
            if isinstance(topic, int) and 0 <= topic < num_topics and isinstance(prob, (int, float)):
                topic_counts_by_year.loc[year, f'Topic {topic}'] += prob
            else:
                print(f"Distribuição de tópico inválida encontrada no ano {year}. Tópico: {topic}, Probabilidade: {prob}")

# Normalizando os valores para percentagem
topic_counts_by_year_percentage = topic_counts_by_year.div(topic_counts_by_year.sum(axis=1), axis=0) * 100

topic_counts_by_year_percentage = topic_counts_by_year_percentage.T
topic_counts_by_year_percentage = topic_counts_by_year_percentage.sort_index(axis=1)


################# styling ##########################################
# CSS styling para o fundo da rede
st.markdown("""
<div style="background-color: black; padding: 20px; border-radius: 5px; text-align: left;">
    <h1 style="font-size: 40px; font-weight: bold; color: #bc5090;">ANÁLISE DE ARTIGOS CIENTÍFICOS</h1>
    <h2 style="font-size: 32px; font-weight: normal; color: #ffa600;">DATA SCIENCE e BIG DATA, aplicadas ao PLANEAMENTO URBANO</h2>
</div>

<style>
/* Estilo do iframe da rede com fundo preto */
iframe {
    background-color: black;
}
</style>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs([" 📈 Análise Bibliométrica", " 📊 Análise de Conteúdo"])

# Aba 1: Análise Bibliográfica
with tab1:
    st.markdown("""
<div style="background-color: #333; padding: 10px; border-radius: 5px; color: white; font-size: 16px; margin-bottom: 20px;">
    Query da pesquisa SCOPUS: Data Science OR Big Data AND (Urban Planning OR Urban Management OR Spatial Planning OR Urban Development), limited to Social Sciences
</div>
""", unsafe_allow_html=True)

    col1, spacer1, col2 = st.columns([3, 0.2, 3])
    # Dividir a tela em três colunas, conforme exemplo fornecido
    with col1:
        with st.container():
            st.subheader('Rede bibliométrica de co-ocorrência de keywords dos autores')
            st.components.v1.html(f'<iframe src="{url}" width="95%" height="500px" style="border: 0px solid #ddd; max-width: 1000px; max-height: 500px; background-color: black;"></iframe>', height=600)
    
    with spacer1:
        st.markdown(' ')
    
    with col2:
        with st.container():
            st.subheader('País de publicação dos artigos')
            st.plotly_chart(fig1, use_container_width=True,
                        width=500, height=500,
                        key="fig1")

    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

    col3, spacer2, col4 = st.columns([3, 0.2, 3])

    with col3:
        st.subheader('Top 10: Termos mais recorrentes nos títulos dos artigos')
        st.plotly_chart(fig, use_container_width=True,
                        width=500, height=500,
                        key="fig")

    with spacer2:
        st.markdown(' ')

    with col4:
        st.subheader('TOP 10: Autores, por nº de publicações')
        st.plotly_chart(fig2,
                        width=500, height=500, key="fig2"
                        )
with tab2:

    # Dividir a tela em três colunas com espaçadores
    col1, spacer1, col2 = st.columns([2, 6, 2])
    with col1:
        st.markdown(' ')
    with spacer1:
        with st.container():
            st.subheader('Palavras mais comuns no abstract dos artigos')
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Remove os eixos
            st.pyplot(plt)
    with col2:
        st.markdown(' ')

    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

# Dividir a tela em três colunas com espaçadores
    col1, spacer1, col2 = st.columns([1, 6, 1])
    with col1:
        st.markdown(' ')
    with spacer1:
        with st.container():
            st.markdown("""
    <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white; font-size: 24px; margin-bottom: 20px;">
        Tópicos gerados, utilizando o modelo LDA:
    </div>
    """, unsafe_allow_html=True)
            st.markdown("""
            **Tópico 0:**
            - **Categoria sugerida**: *Data Science and Urban Analysis*
            - Baseado nos termos como "data", "big", "analysis", "systems", "media", e "knowledge", este cluster parece se concentrar no uso de ciência de dados, métodos computacionais e análise de grandes volumes de informação aplicados a contextos urbanos.

            **Tópico 1:**
            - **Categoria sugerida**: *Spatial and Population Dynamics*
            - Os termos como "spatial", "population", "distribution", "network", "interaction", e "structure" indicam um foco em estudos de dinâmica espacial, população e estrutura urbana, incluindo redes e interações entre áreas urbanas e industriais.

            **Tópico 2:**
            - **Categoria sugerida**: *Smart and Sustainable Cities*
            - Termos como "smart", "cities", "sustainable", "digital", "technologies", e "governance" sugerem que este cluster trata de cidades inteligentes, desenvolvimento sustentável, uso de tecnologias digitais e governança urbana moderna.

            **Tópico 3:**
            - **Categoria sugerida**: *Urban Land Use and Activity Patterns*
            - Com termos como "land", "vitality", "spatial", "travel", "activity", e "patterns", este cluster parece abordar padrões de uso do solo, mobilidade, e a vitalidade ou vibrância das áreas urbanas.
            """)
    with col2:
        st.markdown(' ')

    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")


    # Segunda linha de colunas
    col3, spacer2, col4 = st.columns([3, 0.2, 3])

    with col3:
        st.subheader('Distribuição de Tópicos por País (%)')
        plt.style.use('dark_background')
            # Gerando o gráfico do heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_counts_percentage, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)

        # Alterando título e labels para garantir boa visibilidade em fundo escuro
        plt.xlabel("Tópicos", fontsize=12, color='white')
        plt.ylabel("Países", fontsize=12, color='white')

        # Exibindo o gráfico no Streamlit
        st.pyplot(plt)

    with spacer2:
        st.markdown(' ')

    with col4:
        st.subheader('Distribuição de Tópicos por Ano de Publicação (%)')
        # Visualizando o heatmap com os valores em percentagem
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 8))
        sns.heatmap(topic_counts_by_year_percentage, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)
        plt.xlabel("Tópicos")
        plt.ylabel("Ano de Publicação")

         # Alterando título e labels para garantir boa visibilidade em fundo escuro
        plt.xlabel("Ano de publicação", fontsize=12, color='white')
        plt.ylabel("Tópicos", fontsize=12, color='white')

        st.pyplot(plt)

    col1, spacer1, col2 = st.columns([1, 6, 1])
    with col1:
        st.markdown(' ')
    with spacer1:
        st.subheader('Técnicas de Data Science mais comuns por tópico:')
        top_10_techniques_by_cluster.rename(columns={
        'Topic_Cluster': 'Tópico', 
        'tecnicas_str_categorizadas': 'Técnicas',
        'percentage': '%'
        # Adicione outras colunas que você deseja renomear aqui
 }, inplace=True)
        cluster_choice = st.selectbox("Escolha um Topic Cluster para ver as técnicas mais comuns", top_10_techniques_by_cluster['Tópico'].unique())
        filtered_df = top_10_techniques_by_cluster[top_10_techniques_by_cluster['Tópico'] == cluster_choice]

        st.write(f"Técnicas mais comuns no tópico {cluster_choice}:")
        st.dataframe(filtered_df)
    with col2:
        st.markdown(' ')
        st.dataframe(filtered_df)
    with col2:
        st.markdown(' ')
