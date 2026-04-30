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
pip install numpy pandas matplotlib seaborn scikit-learn graphviz shap
`

**requirements.txt sugerido:**

`
numpy
pandas
matplotlib
seaborn
scikit-learn
graphviz
shap
`

O dataset e carregado via sklearn.datasets.load_breast_cancer(), sem necessidade de download externo.

---

## Como Executar

### 1. Clone o repositorio

`ash
git clone https://github.com/JefersonFurtado/IA_TECH_FIAP.git
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
| 20 | Feature Importance: comparativo Decision Tree vs Random Forest |
| 21 | SHAP TreeExplainer: Summary Plot (bar) e Beeswarm Plot |
| 22 | SHAP Waterfall: analise de predicoes individuais (maligno e benigno) |

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

## Explicabilidade do Modelo — Feature Importance e SHAP

### 1. Feature Importance (Impureza de Gini)

A **importancia baseada em Gini** mede o quanto cada feature contribui para a reducao da impureza nas arvores de decisao. Valores mais altos indicam features mais discriminativas.

#### Decision Tree
Na arvore de decisao isolada, a feature **`worst concave points`** domina com importancia proxima de ~72%, revelando uma dependencia excessiva de uma unica variavel. Isso e um indicativo de **overfitting parcial** — o modelo aprendeu a se apoiar quase exclusivamente nessa feature, o que pode ser fragil em dados nao vistos.

#### Random Forest
No Random Forest, a importancia e distribuida de forma muito mais equilibrada entre multiplas features:
- **`worst concave points`** e **`worst area`**: as duas principais, relacionadas a forma e tamanho do tumor na regiao mais irregular
- **`mean concavity`** e **`worst perimeter`**: complementam a analise de forma
- Features do grupo `mean` e `worst` dominam, enquanto as do grupo `error` tem menor relevancia

Essa distribuicao mais uniforme reflete a **robustez do ensemble** — ao combinar 100 arvores cada uma treinada em subconjuntos aleatorios de features, o Random Forest forcou o modelo a explorar combinacoes variadas, resultando em uma visao mais realista do poder discriminativo de cada variavel.

**Limitacao importante:** A importancia por Gini pode ser tendenciosa para features com mais valores unicos (como areas numericas continuas) e nao captura interacoes entre features. O SHAP resolve essas limitacoes.

---

### 2. SHAP (SHapley Additive exPlanations)

O SHAP usa a teoria dos jogos cooperativos (Valores de Shapley) para atribuir a cada feature uma contribuicao justa e consistente para a predicao de cada amostra individual. Diferente da Feature Importance, o SHAP:
- e **local**: explica predicoes individuais
- e **consistente**: garante que se uma feature contribui mais, seu valor SHAP sera maior
- **leva em conta interacoes** entre features
- mostra a **direcao** da influencia (valores positivos empurram para uma classe, negativos para outra)

#### Summary Plot (Bar)
Confirma a hierarquia da Feature Importance, mas com uma perspectiva global mais confiavel. As features `worst concave points`, `worst area` e `mean concavity` aparecem consistentemente como as mais impactantes para classificar como Benigno.

#### Beeswarm Plot (Summary Dot)
O beeswarm revela insights cruciais alem da importancia:
- **Cores quentes (vermelho)** = valores altos da feature; **Cores frias (azul)** = valores baixos
- Features como `worst concave points` e `worst area`: valores **ALTOS** empurram a predicao para **Maligno**, enquanto valores **BAIXOS** empurram para **Benigno** — coerente com a biologia: tumores com maior area e concavidade tendem a ser malignos
- A largura da distribuicao indica a variabilidade do impacto da feature no conjunto de teste

#### Waterfall Plot (Predicoes Individuais)
O waterfall mostra como o modelo chegou a uma predicao especifica, partindo do valor esperado base e adicionando/subtraindo contribuicoes de cada feature:
- **Barras vermelhas**: features que aumentam a probabilidade da classe analisada
- **Barras azuis**: features que diminuem a probabilidade da classe analisada
- Para um caso **maligno**: features como `worst concave points` alto e `worst area` alto contribuem fortemente para a predicao de malignidade
- Para um caso **benigno**: valores baixos nessas mesmas features reduzem a probabilidade de malignidade

---

### 3. Concordancia entre Feature Importance e SHAP

| Feature | Feature Importance (RF) | SHAP Rank |
|---|---|---|
| `worst concave points` | Top 1-2 | Top 1-2 |
| `worst area` | Top 1-2 | Top 1-2 |
| `mean concavity` | Top 3-5 | Top 3-5 |
| `worst perimeter` | Top 3-5 | Top 3-5 |

Essa concordancia aumenta a **confianca na interpretacao** do modelo: as features identificadas nao sao artefatos de um metodo especifico, mas sim sinais reais do poder discriminativo das variaveis no diagnostico de cancer.

### 4. Implicacoes Clinicas

As features mais importantes correspondem a caracteristicas morfologicas conhecidas na oncologia:
- **`worst concave points`** e **`worst concavity`**: irregularidades no contorno celular sao marcadores classicos de malignidade (celulas malignas perdem a forma regular)
- **`worst area`** e **`worst perimeter`**: tumores malignos geralmente tem celulas maiores
- O prefixo `worst` captura o **pior caso** observado na amostra, o que e clinicamente relevante pois a gravidade de um tumor e frequentemente determinada pelas celulas mais atipicas

O modelo e, portanto, **clinicamente interpretavel**: suas decisoes se baseiam em caracteristicas biologicamente plausíveis, o que e fundamental para a adocao em contextos medicos reais.

---

## Metricas de Avaliacao

- Acuracia: proporcao de predicoes corretas
- F1-score: media harmonica entre Precision e Recall
- Matriz de Confusao: visualizacao dos acertos e erros por classe

---

## Autores

Jeferson Furtado da Silva RM373579

Desenvolvido como parte do Tech Challenge de IA aplicada a area da saude.
