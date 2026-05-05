# Tech Challenge - IA Diagnóstico de Câncer

Projeto de classificação de tumores mamários (benigno ou maligno) utilizando Machine Learning com os modelos Decision Tree e Random Forest, aplicados ao dataset Breast Cancer Wisconsin disponível na biblioteca scikit-learn.

---

## Descrição do Projeto

Este projeto realiza uma análise exploratória completa do dataset e treina dois modelos de classificação supervisionada para auxiliar no diagnóstico de câncer de mama, comparando o desempenho entre ambos.

**Resultados obtidos:**

| Modelo | Acurácia | F1 (Maligno) | F1 (Benigno) | Falsos Negativos |
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

## Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab

---

## Instalação das Dependências

### 1. Crie e ative um ambiente virtual

`ash
python -m venv venv
venv\Scriptsctivate
`

### 2. Instale as bibliotecas

`ash
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

O dataset é carregado via sklearn.datasets.load_breast_cancer(), sem necessidade de download externo.

---

## Como Executar

### 1. Clone o repositório

`ash
git clone https://github.com/JefersonFurtado/IA_TECH_FIAP.git
cd tech-challenge-cancer
`

### 2. Inicie o Jupyter

`ash
jupyter notebook
`

### 3. Abra o notebook

Clique em techChalleng.ipynb no navegador.

### 4. Execute todas as células

Menu: Kernel > Restart and Run All

Ou use Shift+Enter para executar célula por célula.

---

## Estrutura do Notebook

| Célula | Descrição |
|---|---|
| 1 | Importação das bibliotecas |
| 2-6 | Carregamento e preparação do dataset |
| 7-11 | Exploração e validação dos dados |
| 12 | Divisão treino/teste 80/20 |
| 13 | Mapa de correlação (Heatmap) |
| 14 | Estatísticas descritivas |
| 15 | Padronização com StandardScaler |
| 16-17 | Análise de distribuição das features |
| 18 | Decision Tree: treinamento e avaliação |
| 19 | Random Forest: treinamento, avaliação e comparativo |
| 20 | Feature Importance: comparativo Decision Tree vs Random Forest |
| 21 | SHAP TreeExplainer: Summary Plot (bar) e Beeswarm Plot |
| 22 | SHAP Waterfall: análise de predições individuais (maligno e benigno) |

---

## Sobre o Dataset

- Nome: Breast Cancer Wisconsin (Diagnostic)
- Fonte: sklearn.datasets.load_breast_cancer()
- Amostras: 569
- Features: 30 atributos numéricos
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

A **importância baseada em Gini** mede o quanto cada feature contribui para a redução da impureza nas árvores de decisão. Valores mais altos indicam features mais discriminativas.

#### Decision Tree
Na árvore de decisão isolada, a feature **`worst concave points`** domina com importância próxima de ~72%, revelando uma dependência excessiva de uma única variável. Isso é um indicativo de **overfitting parcial** — o modelo aprendeu a se apoiar quase exclusivamente nessa feature, o que pode ser frágil em dados não vistos.

#### Random Forest
No Random Forest, a importância é distribuída de forma muito mais equilibrada entre múltiplas features:
- **`worst concave points`** e **`worst area`**: as duas principais, relacionadas à forma e tamanho do tumor na região mais irregular
- **`mean concavity`** e **`worst perimeter`**: complementam a análise de forma
- Features do grupo `mean` e `worst` dominam, enquanto as do grupo `error` têm menor relevância

Essa distribuição mais uniforme reflete a **robustez do ensemble** — ao combinar 100 árvores cada uma treinada em subconjuntos aleatórios de features, o Random Forest forçou o modelo a explorar combinações variadas, resultando em uma visão mais realista do poder discriminativo de cada variável.

**Limitação importante:** A importância por Gini pode ser tendenciosa para features com mais valores únicos (como áreas numéricas contínuas) e não captura interações entre features. O SHAP resolve essas limitações.

---

### 2. SHAP (SHapley Additive exPlanations)

O SHAP usa a teoria dos jogos cooperativos (Valores de Shapley) para atribuir a cada feature uma contribuição justa e consistente para a predição de cada amostra individual. Diferente da Feature Importance, o SHAP:
- é **local**: explica predições individuais
- é **consistente**: garante que se uma feature contribui mais, seu valor SHAP será maior
- **leva em conta interações** entre features
- mostra a **direção** da influência (valores positivos empurram para uma classe, negativos para outra)

#### Summary Plot (Bar)
Confirma a hierarquia da Feature Importance, mas com uma perspectiva global mais confiável. As features `worst concave points`, `worst area` e `mean concavity` aparecem consistentemente como as mais impactantes para classificar como Benigno.

#### Beeswarm Plot (Summary Dot)
O beeswarm revela insights cruciais além da importância:
- **Cores quentes (vermelho)** = valores altos da feature; **Cores frias (azul)** = valores baixos
- Features como `worst concave points` e `worst area`: valores **ALTOS** empurram a predição para **Maligno**, enquanto valores **BAIXOS** empurram para **Benigno** — coerente com a biologia: tumores com maior área e concavidade tendem a ser malignos
- A largura da distribuição indica a variabilidade do impacto da feature no conjunto de teste

#### Waterfall Plot (Predições Individuais)
O waterfall mostra como o modelo chegou a uma predição específica, partindo do valor esperado base e adicionando/subtraindo contribuições de cada feature:
- **Barras vermelhas**: features que aumentam a probabilidade da classe analisada
- **Barras azuis**: features que diminuem a probabilidade da classe analisada
- Para um caso **maligno**: features como `worst concave points` alto e `worst area` alto contribuem fortemente para a predição de malignidade
- Para um caso **benigno**: valores baixos nessas mesmas features reduzem a probabilidade de malignidade

---

### 3. Concordância entre Feature Importance e SHAP

| Feature | Feature Importance (RF) | SHAP Rank |
|---|---|---|
| `worst concave points` | Top 1-2 | Top 1-2 |
| `worst area` | Top 1-2 | Top 1-2 |
| `mean concavity` | Top 3-5 | Top 3-5 |
| `worst perimeter` | Top 3-5 | Top 3-5 |

Essa concordância aumenta a **confiança na interpretação** do modelo: as features identificadas não são artefatos de um método específico, mas sim sinais reais do poder discriminativo das variáveis no diagnóstico de câncer.

### 4. Implicações Clínicas

As features mais importantes correspondem a características morfológicas conhecidas na oncologia:
- **`worst concave points`** e **`worst concavity`**: irregularidades no contorno celular são marcadores clássicos de malignidade (células malignas perdem a forma regular)
- **`worst area`** e **`worst perimeter`**: tumores malignos geralmente têm células maiores
- O prefixo `worst` captura o **pior caso** observado na amostra, o que é clinicamente relevante pois a gravidade de um tumor é frequentemente determinada pelas células mais atípicas

O modelo é, portanto, **clinicamente interpretável**: suas decisões se baseiam em características biologicamente plausíveis, o que é fundamental para a adoção em contextos médicos reais.

---

## Métricas de Avaliação

- Acurácia: proporção de predições corretas
- F1-score: média harmônica entre Precision e Recall
- Matriz de Confusão: visualização dos acertos e erros por classe

---

## Autores

Jeferson Furtado da Silva RM373579
Najla Peixoto Pinheiro RM373765

Desenvolvido como parte do Tech Challenge de IA aplicada à área da saúde.
