# 📊 Painel de Análise de Sentimentos e Bigramas — Swift

Aplicação interativa em **Streamlit** para análise de comentários, extração de bigramas, acompanhamento de NPS, categorização e visualização temporal da evolução do sentimento dos clientes.

O painel foi desenvolvido para transformar um arquivo CSV contendo avaliações de clientes em dashboards interativos que auxiliam na tomada de decisões operacionais e estratégicas.

---

## 🚀 Funcionalidades

O sistema oferece diversas ferramentas analíticas:

### ✔️ Visão Geral
- Métricas gerais (total de avaliações, promotores, neutros e detratores)  
- Distribuição de NPS (gráfico de pizza)  
- Categorias mais comentadas (gráfico de barras)

### ✔️ Análise de Bigramas
- Bigramas mais frequentes por NPS  
- Bigramas por categoria  
- Bigramas por sentimento  
- Bigramas negativos por categoria (para insights operacionais)

### ✔️ Categorias & NPS
- Tabela dinâmica (pivot) mostrando volume por categoria  
- Categorias com mais detratores  
- Categorias com mais promotores  
- Distribuições positivas/negativas por categoria

### ✔️ Séries Temporais
- Evolução do sentimento ao longo dos meses  
- Comparação entre detratores e promotores ao longo do tempo

### ✔️ Exportação e Ações Recomendadas
Exportação de:
- **Top bigramas por NPS**  
- **Pivot categorias x NPS**

Geração automática de ações recomendadas com base nos bigramas negativos.

---

## 🗂️ Estrutura do Projeto
├── app.py
├── Comentarios.csv # Arquivo de entrada
├── stopwords.txt # Stopwords personalizadas
├── swift-convertido-de-jpeg (1).png
└── README.md


---

## 🧠 Requisitos

### Python 3.9+

### Bibliotecas

Instale com:

```bash
pip install -r requirements.txt
```
## 🧠 Principais Dependências

As principais bibliotecas utilizadas pelo projeto são:

- **streamlit**
- **pandas**
- **numpy**
- **plotly**
- **scikit-learn**
- **nltk**

---

## 📥 Formato do CSV (Obrigatório)

O arquivo **Comentarios.csv** deve conter obrigatoriamente as seguintes colunas:

| Coluna          | Descrição                           |
|-----------------|--------------------------------------|
| **Ano**         | Ano da avaliação                     |
| **Centro**      | Loja / unidade                       |
| **Mes**         | Mês numérico                         |
| **Semana**      | Número da semana                     |
| **Data Avaliação** | Data completa da avaliação       |
| **NPS**         | Promotor / Neutro / Detrator         |
| **Categoria**   | Categoria relacionada ao comentário   |
| **final_text**  | Texto original do comentário          |

A aplicação valida automaticamente a existência dessas colunas.

---

## ▶️ Como Executar

1. Coloque o arquivo **Comentarios.csv** na pasta do projeto.  
2. Execute o comando:

```bash
streamlit run app.py
```

## 🧹 Limpeza e Normalização de Texto

A aplicação realiza automaticamente:

- **lowercasing**
- **remoção de links**
- **remoção de números**
- **remoção de caracteres especiais**
- **remoção de tokens com apenas 1 letra**
- **aplicação de stopwords**
- **agrupamento e normalização de tokens com base no vocabulário tratado**

**Stopwords utilizadas:**
- `stopwords.txt`
- conjunto `CUSTOM_STOPWORDS` dentro do código

---

## 🔍 Bigramas

Bigramas são gerados utilizando **CountVectorizer**, com:

- ngramas de **2 palavras**
- filtragem de **stopwords**
- exclusão de **palavras de 1 letra**
- exclusão de **tokens contendo números**
- **ordenação por frequência**

---

## 🎨 Interface Dark Customizada

O projeto possui um tema escuro completo com CSS customizado:

- Sidebar personalizada  
- Tabs estilizadas  
- Métricas com fundo escuro  
- Paleta compatível com gráficos Plotly  
- Layout totalmente responsivo  

---

## 📤 Exportação

O painel permite exportar automaticamente:

- `top_bigrams_by_nps.csv`
- `pivot_categoria_nps.csv`

Os arquivos são gerados automaticamente na pasta do projeto.

---

## 🧭 Ações Automáticas Recomendadas

A aba final do painel sugere ações com base nos **bigramas negativos mais frequentes**, como:

- Revisão de estoque  
- Padronização de respostas de atendimento  
- Correções de atrasos  
- Monitoramento de padrões negativos  
- Ações de comunicação com base em pontos fortes  

---

## 🛠️ Melhorias Futuras (Sugestões)

- Classificação de sentimento via NLP (transformers)  
- Filtros por cidade/região  
- Dashboard de SLA de atendimento + NPS  
- Upload dinâmico de novos CSVs  
- Versão em Docker  

---

## 🤝 Contribuições

Sinta-se à vontade para abrir **issues** ou enviar **pull requests**.