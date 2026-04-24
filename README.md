# Tech Challenge - IA Diagnostico de Cancer

Projeto de classificacao de tumores mamarios (benigno ou maligno) utilizando Machine Learning com os modelos Decision Tree e Random Forest, aplicados ao dataset Breast Cancer Wisconsin disponivel na biblioteca scikit-learn.

---

## Descricao do Projeto

Este projeto realiza uma analise exploratoria completa do dataset e treina dois modelos de classificacao supervisionada para auxiliar no diagnostico de cancer de mama, comparando o desempenho entre ambos.

**Resultados obtidos:**

| Modelo | Acuracia | F1 (Maligno) | F1 (Benigno) | Falsos Negativos |
|---|---|---|---|---|
| Decision Tree | 95.61% | 0.93 | 0.97 | 2 |
| Random Forest | 98.25% | 0.97 | 0.99 | 0 |

---

## Estrutura do Projeto

`
techChalleng/
├── techChalleng.ipynb
└── README.md
`

---

## Pre-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab

---

## Instalacao das Dependencias

### 1. Crie e ative um ambiente virtual

`ash
python -m venv venv
venv\Scriptsctivate
`

### 2. Instale as bibliotecas

`ash
pip install numpy pandas matplotlib seaborn scikit-learn graphviz
`

**requirements.txt sugerido:**

`
numpy
pandas
matplotlib
seaborn
scikit-learn
graphviz
`

O dataset e carregado via sklearn.datasets.load_breast_cancer(), sem necessidade de download externo.

---

## Como Executar

### 1. Clone o repositorio

`ash
git clone https://github.com/seu-usuario/tech-challenge-cancer.git
cd tech-challenge-cancer
`

### 2. Inicie o Jupyter

`ash
jupyter notebook
`

### 3. Abra o notebook

Clique em techChalleng.ipynb no navegador.

### 4. Execute todas as celulas

Menu: Kernel > Restart and Run All

Ou use Shift+Enter para executar celula por celula.

---

## Estrutura do Notebook

| Celula | Descricao |
|---|---|
| 1 | Importacao das bibliotecas |
| 2-6 | Carregamento e preparacao do dataset |
| 7-11 | Exploracao e validacao dos dados |
| 12 | Divisao treino/teste 80/20 |
| 13 | Mapa de correlacao (Heatmap) |
| 14 | Estatisticas descritivas |
| 15 | Padronizacao com StandardScaler |
| 16-17 | Analise de distribuicao das features |
| 18 | Decision Tree: treinamento e avaliacao |
| 19 | Random Forest: treinamento, avaliacao e comparativo |

---

## Sobre o Dataset

- Nome: Breast Cancer Wisconsin (Diagnostic)
- Fonte: sklearn.datasets.load_breast_cancer()
- Amostras: 569
- Features: 30 atributos numericos
- Classes: Maligno (0) 212 amostras, Benigno (1) 357 amostras
- Valores nulos: Nenhum

---

## Modelos Utilizados

### Decision Tree
- criterion=gini, max_depth=5, random_state=42

### Random Forest
- n_estimators=100, criterion=gini, max_depth=5, random_state=42

---

## Metricas de Avaliacao

- Acuracia: proporcao de predicoes corretas
- F1-score: media harmonica entre Precision e Recall
- Matriz de Confusao: visualizacao dos acertos e erros por classe

---

## Autor

Desenvolvido como parte do Tech Challenge de IA aplicada a area da saude.