
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
from sklearn.datasets import load_breast_cancer

# Carrega base de dados de câncer de mama do sklearn
cancer = load_breast_cancer()

# concatenar arrays de dados e rótulos para criar um DataFrame
data = np.c_[cancer.data, cancer.target]

# criar lista de nomes de colunas para o DataFrame
columns = list(cancer.feature_names) + [np.str_("target")]

#Verifica se tem alguma coluna sem nome
print("Nomes das colunas: ", columns)

#criar DataFrame usando os dados e os nomes das colunas
df = pd.DataFrame(data, columns=columns)

#printar as primeiras linhas do DataFrame para verificar os dados
print(df.head())

# verificar se há valores ausentes no DataFrame
print(df.isna().sum())

# verificar se há valores nulos no DataFrame
print(df.isnull().sum())


#criando modelo de classificação para prever se um tumor é maligno ou benigno
y = df.target
X = df[df.columns[:-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)


#visualização da distribuição dos rótulos de classe
ax = sns.countplot(y,label="Count")       # M = 212, B = 357
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)

print(X.describe())