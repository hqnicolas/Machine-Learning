# Aula 7 – Floresta Aleatória

---

## O que é Árvore de Decisão?

É um tipo de algoritmo de aprendizagem de máquina supervisionado.

Se baseia na ideia de divisão dos dados em grupos homogêneos, podendo ser utilizadas em um cenário de classificação ou regressão.

<img width="936" height="454" alt="image" src="https://github.com/user-attachments/assets/c61de19c-6a42-415a-a0d7-e08c709fa709" />

---

## Floresta Aleatória – Random Forest

Floresta Aleatória (random forest) é um algoritmo de aprendizagem supervisionada.

A "floresta" que ele cria é uma combinação (ensemble) de árvores de decisão, na maioria dos casos treinados com o método de bagging. A ideia principal do método de bagging é que a combinação dos modelos de aprendizado aumenta o resultado geral.

---

### Funcionamento

<img width="1383" height="818" alt="image" src="https://github.com/user-attachments/assets/3a1d05c5-8507-4e23-b531-cb734a733bb7" />

---

### Regressão
<img width="1052" height="729" alt="image" src="https://github.com/user-attachments/assets/ac4fa298-cdb7-4234-9fab-08e81584e499" />


---

### Classificação
<img width="1087" height="694" alt="image" src="https://github.com/user-attachments/assets/6c483b9e-fcc2-49b3-8da1-836c4f897bc6" />


---

## Hiperparâmetros

* **Número de árvores (n_estimators):** número de árvores construídas pelo algoritmo antes de tomar uma votação ou fazer uma média de predições.
* **Número de max características (max_features):** indica o número máximo de características a serem utilizadas pelo Floresta Aleatória na construção de uma dada árvore.
* **Número de min folhas (min_sample_leaf):** indica o número mínimo de folhas que devem existir em uma dada árvore.
* **n_Jobs:** informa quantos processadores o algoritmo pode utilizar. Se ele tiver valor 1, pode utilizar apenas um processador.
* **oob_score:** (também chamado de oob sampling), que é um método de validação cruzada para floresta aleatória.

---

## Praticando Machine Learning – Dataset ‘penguins’

### Importando e Listando Datasets

```

# Importando a lib do seaborn para pegar um dataset

import seaborn

# Listando os datasets disponíveis.

seaborn.get\_dataset\_names()

```

Saída:

```

['anagrams', 'anscombe', 'attention', 'brain\_networks', 'car\_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'geyser', 'iris', 'mpg', 'penguins', 'planets', 'tips', 'titanic']

```

### Carregando o Dataset

```

# Carregando o dataset dos pinguins.

dataframe = seaborn.load\_dataset('penguins')

# Visualização resumida.

dataframe.head()

```

### Espécies no Dataset

```

species\_names = pd.unique(dataframe['species'])
species\_names

```

Saída:

```

array(['Adelie', 'Chinstrap', 'Gentoo'], dtype=object)

```

| Adelie | Chinstrap | Gentoo |
| :----: | :-------: | :----: |
|        |           |        |

### Análise de Dados

```

# Visualização das informações das variáveis.

dataframe.info()

```

```

# Verificando os dados nulos.

dataframe.isnull().sum()

```

### Pré-processamento

```

# Importando o Ordinal encoder para transformação

from sklearn.preprocessing import OrdinalEncoder

# Instânciando o modelo.

model\_ordinal = OrdinalEncoder()

# Verificando os nomes das colunas do tipo object

dataframe.select\_dtypes(['object']).columns

```

Saída:

```

Index(['species', 'island', 'sex'], dtype='object')

```

```

# Realizando a transformação das variáveis cat para num

tranf\_values = model\_ordinal.fit\_transform(dataframe.select\_dtypes(['object']))

# Substituindo os valores.

dataframe[dataframe.select\_dtypes(['object']).columns] = tranf\_values

```

```

# Substituindo os dados nulos pela média.

dataframe.fillna(dataframe.mean(),inplace=True)

# Verificando novamente os dados nulos.

dataframe.isnull().sum()

```

```

# Deletando as linhas que restaram nulas.

dataframe.dropna(inplace=True)

```

### Divisão dos Dados

```

# Pegando as variáveis de entrada do modelo

x = dataframe.drop(['species'],axis=1)

```

```

# Pegando a variável de saída do modelo

y = dataframe['species']

```

```

# importando a lib para validação cruzada.

from sklearn.model\_selection import train\_test\_split

# Separando os dados em treino e teste.

x\_train,x\_test, y\_train, y\_test = train\_test\_split(x,y,test\_size=0.20)

```

### Treinamento do Modelo

```

# Importando a lib da floresta aleatória para classificação.

from sklearn.ensemble import RandomForestClassifier

model\_random = RandomForestClassifier(n\_estimators=5,max\_depth=6)

model\_random.fit(x\_train,y\_train)

```

### Predição

```

# Realizando predição.

y\_predict = model\_random.predict(x\_test)

# Pedindo a predição dos valores probabilístico.

y\_predict\_prob = model\_random.predict\_proba(x\_test)

y\_predict\_prob[3]

```

Saída:

```

array([0.17101449, 0.82898551, 0.        ])

```

### Análise do Modelo

```

model\_random.estimators\_[3]

```

### Visualização da Árvore

```

# Importando as libs para exportar a arquitetura da figura.

from sklearn.tree import plot\_tree,export\_graphviz
import graphviz

# Utilizando o plot simples

plot\_tree(model\_random.estimators\_[3],
feature\_names=x.columns,
class\_names=species\_names,
filled=True)

```

```

dot\_data = export\_graphviz(model\_random.estimators\_[3],
feature\_names=x.columns,
class\_names=species\_names,
filled=True)

graph = graphviz.Source(dot\_data, format="png")
graph

```

-----

## Exemplo Floresta Aleatória - Regressão

```

> > > from sklearn.ensemble import RandomForestRegressor
> > > from sklearn.datasets import make\_regression
> > > X, y = make\_regression(n\_features=4, n\_informative=2,
> > > ...                        random\_state=0, shuffle=False)
> > > regr = RandomForestRegressor(max\_depth=2, random\_state=0)
> > > regr.fit(X, y)
> > > RandomForestRegressor(...)
> > > print(regr.predict([[0, 0, 0, 0]]))
> > > [-8.32987858]

```

-----

## Exercícios – Lista 2

Escolha um conjunto de dados que permita para tarefa de classificação ou regressão:

1.  Treine um modelo de Árvore de Decisão para resolver cada tarefa e registre os resultados(métricas de desempenho).
2.  Em seguida, treine um modelo de Floresta Randômica (Random Forest) com os mesmos dados.
3.  Compare os resultados obtidos, mostrando que, em geral, a Floresta Randômica apresenta melhor desempenho e maior capacidade de generalização do que uma única Árvore de Decisão.
4. Prove que uma Floresta de decisão é mais eficiente do que uma árvore de decisão
