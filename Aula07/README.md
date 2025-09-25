````markdown
# [cite_start]Aula 7 – Floresta Aleatória [cite: 1]

[cite_start]**Autor:** Yuri Crotti [cite: 2]

---

## [cite_start]O que é Árvore de Decisão? [cite: 3]

[cite_start]É um tipo de algoritmo de aprendizagem de máquina supervisionado[cite: 5].

[cite_start]Se baseia na ideia de divisão dos dados em grupos homogêneos, podendo ser utilizadas em um cenário de classificação ou regressão[cite: 6].

![Exemplo de Árvore de Decisão para Regressão](https://storage.googleapis.com/generativeai-downloads/images/05e19e075051a1157e3f053597d519b7)

![Exemplo de Árvore de Decisão para Classificação](https://storage.googleapis.com/generativeai-downloads/images/d378297b5e40632b49231f452e257a07)

---

## [cite_start]Floresta Aleatória – Random Forest [cite: 7]

[cite_start]Floresta Aleatória (random forest) é um algoritmo de aprendizagem supervisionada[cite: 9].

[cite_start]A "floresta" que ele cria é uma combinação (ensemble) de árvores de decisão, na maioria dos casos treinados com o método de bagging[cite: 10]. [cite_start]A ideia principal do método de bagging é que a combinação dos modelos de aprendizado aumenta o resultado geral[cite: 11].

---

### [cite_start]Funcionamento [cite: 12]

![Diagrama do Funcionamento do Random Forest](https://storage.googleapis.com/generativeai-downloads/images/50f90033c5e88414400e9603f7d13eb7)

---

### [cite_start]Regressão [cite: 14]

![Diagrama de Regressão com Random Forest](https://storage.googleapis.com/generativeai-downloads/images/3fc179a32c2a05786dd6312411e7372d)

---

### [cite_start]Classificação [cite: 16]

![Diagrama de Classificação com Random Forest](https://storage.googleapis.com/generativeai-downloads/images/0c069b16867e163b82736186b51c142d)

---

## [cite_start]Hiperparâmetros [cite: 18]

* [cite_start]**Número de árvores (n_estimators):** número de árvores construídas pelo algoritmo antes de tomar uma votação ou fazer uma média de predições[cite: 20].
* [cite_start]**Número de max características (max_features):** indica o número máximo de características a serem utilizadas pelo Floresta Aleatória na construção de uma dada árvore[cite: 21].
* [cite_start]**Número de min folhas (min_sample_leaf):** indica o número mínimo de folhas que devem existir em uma dada árvore[cite: 22].
* **n_Jobs:** informa quantos processadores o algoritmo pode utilizar. [cite_start]Se ele tiver valor 1, pode utilizar apenas um processador[cite: 23].
* [cite_start]**oob_score:** (também chamado de oob sampling), que é um método de validação cruzada para floresta aleatória[cite: 24].

---

## [cite_start]Praticando Machine Learning – Dataset ‘penguins’ [cite: 25]

### [cite_start]Importando e Listando Datasets [cite: 25]

```python
# Importando a lib do seaborn para pegar um dataset
import seaborn

# Listando os datasets disponíveis.
seaborn.get_dataset_names()
````

Saída:

```
['anagrams', 'anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 'gammas', 'geyser', 'iris', 'mpg', 'penguins', 'planets', 'tips', 'titanic']
```

### [cite\_start]Carregando o Dataset [cite: 27]

```python
# Carregando o dataset dos pinguins.
dataframe = seaborn.load_dataset('penguins')

# Visualização resumida.
dataframe.head()
```

### [cite\_start]Espécies no Dataset [cite: 29]

```python
species_names = pd.unique(dataframe['species'])
species_names
```

Saída:

```
array(['Adelie', 'Chinstrap', 'Gentoo'], dtype=object)
```

| Adelie | Chinstrap | Gentoo |
| :---: |:---:|:---:|
|  |  |  |

### [cite\_start]Análise de Dados [cite: 31]

```python
# Visualização das informações das variáveis.
dataframe.info()
```

```python
# Verificando os dados nulos.
dataframe.isnull().sum()
```

### [cite\_start]Pré-processamento [cite: 33]

```python
# Importando o Ordinal encoder para transformação
from sklearn.preprocessing import OrdinalEncoder

# Instânciando o modelo.
model_ordinal = OrdinalEncoder()

# Verificando os nomes das colunas do tipo object
dataframe.select_dtypes(['object']).columns
```

Saída:

```
Index(['species', 'island', 'sex'], dtype='object')
```

```python
# Realizando a transformação das variáveis cat para num
tranf_values = model_ordinal.fit_transform(dataframe.select_dtypes(['object']))

# Substituindo os valores.
dataframe[dataframe.select_dtypes(['object']).columns] = tranf_values
```

```python
# Substituindo os dados nulos pela média.
dataframe.fillna(dataframe.mean(),inplace=True)

# Verificando novamente os dados nulos.
dataframe.isnull().sum()
```

```python
# Deletando as linhas que restaram nulas.
dataframe.dropna(inplace=True)
```

### [cite\_start]Divisão dos Dados [cite: 39]

```python
# Pegando as variáveis de entrada do modelo
x = dataframe.drop(['species'],axis=1)
```

```python
# Pegando a variável de saída do modelo
y = dataframe['species']
```

```python
# importando a lib para validação cruzada.
from sklearn.model_selection import train_test_split

# Separando os dados em treino e teste.
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
```

### [cite\_start]Treinamento do Modelo [cite: 45]

```python
# Importando a lib da floresta aleatória para classificação.
from sklearn.ensemble import RandomForestClassifier

model_random = RandomForestClassifier(n_estimators=5,max_depth=6)

model_random.fit(x_train,y_train)
```

### [cite\_start]Predição [cite: 47]

```python
# Realizando predição.
y_predict = model_random.predict(x_test)

# Pedindo a predição dos valores probabilístico.
y_predict_prob = model_random.predict_proba(x_test)

y_predict_prob[3]
```

Saída:

```
array([0.17101449, 0.82898551, 0.        ])
```

### [cite\_start]Análise do Modelo [cite: 49]

```python
model_random.estimators_[3]
```

### [cite\_start]Visualização da Árvore [cite: 51]

```python
# Importando as libs para exportar a arquitetura da figura.
from sklearn.tree import plot_tree,export_graphviz
import graphviz

# Utilizando o plot simples
plot_tree(model_random.estimators_[3],
          feature_names=x.columns,
          class_names=species_names,
          filled=True)
```

```python
dot_data = export_graphviz(model_random.estimators_[3],
                           feature_names=x.columns,
                           class_names=species_names,
                           filled=True)

graph = graphviz.Source(dot_data, format="png")
graph
```

-----

## [cite\_start]Exemplo Floresta Aleatória - Regressão [cite: 55]

```python
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = RandomForestRegressor(max_depth=2, random_state=0)
>>> regr.fit(X, y)
RandomForestRegressor(...)
>>> print(regr.predict([[0, 0, 0, 0]]))
[-8.32987858]
```

-----

## [cite\_start]Exercícios – Lista 2 [cite: 57]

[cite\_start]Escolha um conjunto de dados que permita para tarefa de classificação ou regressão: [cite: 59]

1.  [cite\_start]Treine um modelo de Árvore de Decisão para resolver cada tarefa e registre os resultados(métricas de desempenho)[cite: 60].
2.  [cite\_start]Em seguida, treine um modelo de Floresta Randômica (Random Forest) com os mesmos dados[cite: 61].
3.  [cite\_start]Compare os resultados obtidos, mostrando que, em geral, a Floresta Randômica apresenta melhor desempenho e maior capacidade de generalização do que uma única Árvore de Decisão[cite: 62].

<!-- end list -->

```
```
