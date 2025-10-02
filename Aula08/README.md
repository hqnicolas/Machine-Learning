# Treino, Validação e Teste
## Conceitos de Overfitting e Underfitting

[Concept.ipynb](/Aula08/K-Fold.ipynb)

Underfitting e Overfitting são dois termos extremamente importantes no ramo do machine learning.

<img width="1316" height="473" alt="image" src="https://github.com/user-attachments/assets/87d1a7c8-dabd-4006-bf6b-d48ecafa7b66" />

-----

## Ocorrência de Overfitting

Um cenário de overfitting ocorre quando, nos dados de treino, o modelo tem um desempenho excelente, porém quando utilizamos os dados de teste o resultado é ruim.
É comum ouvirmos que neste cenário o modelo treinado não tem capacidade de generalização.

*A imagem à esquerda mostra um modelo "Overfitted" onde a linha do modelo passa exatamente por cada ponto de dados, capturando o ruído. A imagem à direita compara Under-fitting (uma linha simples que não separa as classes), Appropriate-fitting (uma curva suave que separa bem as classes) e Over-fitting (uma linha complexa que se contorce para classificar cada ponto de treino perfeitamente, mas que provavelmente não generalizará bem).*

-----

## [cite\_start]Ocorrência de Underfitting

Neste cenário o desempenho do modelo já é ruim no próprio treinamento. O modelo não consegue encontrar relações entre as variáveis e o teste nem precisa acontecer. Este modelo já pode ser descartado, pois não terá utilidade.

*A imagem à esquerda mostra um modelo com underfitting, onde uma linha reta simples falha em capturar a tendência curva dos dados. A imagem à direita reitera a comparação entre under-fitting, appropriate-fitting e over-fitting.*

-----

## Underfitting e Overfitting

| | **Underfitting** | **Just right** | **Overfitting** |
| :--- | :--- | :--- | :--- |
| **Sintomas** | • Erro de treinamento elevado<br>• Erro de treinamento próximo ao erro de teste<br>• Viés elevado | • Erro de treinamento ligeiramente menor que erro de teste | • Erro de treinamento muito baixo<br>• Erro de treinamento muito menor que erro de teste<br>• Alta variância |
| **Exemplo de regressão** | *Gráfico com uma linha reta que não se ajusta bem aos pontos de dados.* | *Gráfico com uma curva parabólica que se ajusta bem à tendência dos pontos de dados.* | *Gráfico com uma linha irregular que passa por quase todos os pontos de dados.* |
| **Exemplo de classificação** | *Gráfico com uma linha reta que falha em separar as duas classes de pontos.* | *Gráfico com uma curva suave que separa bem as duas classes de pontos.* | *Gráfico com uma linha muito complexa e irregular que separa perfeitamente os pontos de treino, mas contorna pontos individuais.* |
| **Possíveis remédios** | • Modelo de complexificação<br>• Adicionar mais recursos<br>• Treinar mais | | • Executar a regularização<br>• Obter mais dados |

-----

## Viés e Variância

Em poucas palavras, quando você está procurando um modelo perfeito para explicar seus dados, está buscando equilíbrio entre o viés (bias) e variância (variance).

**Balanço viés/variância:** Quanto mais simples o modelo, maior o viés e, quanto mais complexo o modelo, maior a variância.

*A imagem mostra quatro alvos para ilustrar o conceito de viés e variância:*

  * **Baixo viés, Baixa variância:** Tiros agrupados no centro do alvo (ideal).
  * **Baixo viés, Alta variância:** Tiros espalhados ao redor do centro do alvo.
  * **Alto viés, Baixa variância:** Tiros agrupados, mas fora do centro do alvo.
  * **Alto viés, Alta variância:** Tiros espalhados e fora do centro do alvo.

-----

## Curva de Aprendizado

*A primeira imagem mostra um gráfico da "Curva de Aprendizado" com a capacidade do modelo no eixo x e o erro no eixo y. Ela ilustra que, à medida que a capacidade do modelo aumenta, o viés (bias) diminui enquanto a variância (variance) aumenta. O erro de generalização (generalization error) é a soma de ambos, e a "capacidade ótima" (optimal capacity) é o ponto onde o erro de generalização é mínimo. A área à esquerda é a "zona de underfitting" e à direita é a "zona de overfitting".*

*A segunda imagem mostra um gráfico semelhante com a "Complexidade do modelo" no eixo x e o "Erro de predição" no eixo y. A curva azul representa o erro na "Amostra de treinamento", que diminui com o aumento da complexidade. A curva vermelha representa o erro na "Amostra de validação", que diminui até um ponto e depois começa a aumentar, indicando overfitting.*

-----

## Validação Cruzada (Cross Validation)

Validação cruzada é uma técnica para avaliar modelos de ML por meio de treinamento de vários modelos de ML em subconjuntos de dados de entrada disponíveis e avaliação deles no subconjunto complementar dos dados. O objetivo da validação cruzada é definir um conjunto de dados para "testar" o modelo na fase de treinamento (ou seja, o conjunto de validação), a fim de limitar problemas como overfitting, dar uma visão de como o modelo irá generalizar para um conjunto de dados independente.

**Métodos:**

  * Método Houldout 
  * K-fold
  * Leave One Out
  * Leave P Out
  * SuffleSplit
  * TimeSeries Split

-----

## Método de validação Holdout

Este método consiste em dividir o conjunto total de dados em dois subconjuntos mutuamente exclusivos, um para treinamento (estimação dos parâmetros) e outro para teste (validação). Normalmente se divide em uma proporção 70/30, onde o 70 representa 70% do seu conjunto de dados, que por sua vez será usado para treinar seu modelo e 30 os 30% restantes para testar sua performance.

*Diagrama mostrando um bloco de "Data" sendo dividido em um bloco maior de "Training" e um bloco menor de "Test".*

-----

## Praticando Machine Learning

### Importando as libs básicas

```python
import pandas as pd
import numpy as np
```

### Lendo o csv

```python
dataframe = pd.read_csv('/content/DatasetGen.csv', encoding='latin1', sep=';')
```

### Apresentando o dataframe

```python
dataframe
```

**Output:**

```
	Nome	Sexo	Idade	Cidade	Valor Plano
0	Maria da Silva	F	25	Criciúma	231
1	Leandro Fontanela	M	29	Urussanga	221
2	Gabriel Matias	M	22	Araranguá	125
3	Leonardo Cunha	M	33	Cocal do Sul	150
4	Ana Carolina Souza	F	25	Treviso	200
5	Fernada Ricardo	F	35	Cocal do Sul	335
6	Gustavo Pereira	M	18	Orleans	189
7	Leandro Melo	M	31	Criciúma	184
8	Tiago Cezar	M	25	Araranguá	120
9	Rafael Monteiro	M	29	Criciúma	300
```

### Armazenando as variáveis de entrada

```python
x = dataframe.drop('Valor Plano', axis=1)
x
```

**Output:**

```
	Nome	Sexo	Idade	Cidade
0	Maria da Silva	F	25	Criciúma
1	Leandro Fontanela	M	29	Urussanga
2	Gabriel Matias	M	22	Araranguá
3	Leonardo Cunha	M	33	Cocal do Sul
4	Ana Carolina Souza	F	25	Treviso
5	Fernada Ricardo	F	35	Cocal do Sul
6	Gustavo Pereira	M	18	Orleans
7	Leandro Melo	M	31	Criciúma
8	Tiago Cezar	M	25	Araranguá
9	Rafael Monteiro	M	29	Criciúma
```

### Armazenando a variável de saída

```python
y = dataframe['Valor Plano']
y
```

**Output:**

```
0    231
1    221
2    125
3    150
4    200
5    335
6    189
7    184
8    120
9    300
Name: Valor Plano, dtype: int64
```

-----

## Praticando Método Holdout (Train Test Split)

### Importando a lib e separando os dados

```python
# Importando a lib para train_test_split = método houldout
from sklearn.model_selection import train_test_split

# Separando os dados em treino e teste, de acordo com a % passada.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)
```

### Apresentando as variáveis de treino e teste

**x\_train:**

```
	Nome	Sexo	Idade	Cidade
6	Gustavo Pereira	M	18	Orleans
4	Ana Carolina Souza	F	25	Treviso
0	Maria da Silva	F	25	Criciúma
3	Leonardo Cunha	M	33	Cocal do Sul
1	Leandro Fontanela	M	29	Urussanga
7	Leandro Melo	M	31	Criciúma
8	Tiago Cezar	M	25	Araranguá
5	Fernada Ricardo	F	35	Cocal do Sul
```

**x\_test:**

```
	Nome	Sexo	Idade	Cidade
2	Gabriel Matias	M	22	Araranguá
9	Rafael Monteiro	M	29	Criciúma
```

**y\_train:**

```
6    189
4    200
0    231
3    150
1    221
7    184
8    120
5    335
Name: Valor Plano, dtype: int64
```

**y\_test:**

```
2    125
9    300
Name: Valor Plano, dtype: int64
```

### Praticando com estratificação

```python
# Separando os dados em treino e teste, de acordo com a % passada.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1, stratify=x.Sexo)
```

**x\_train com estratificação:**

```
	Nome	Sexo	Idade	Cidade
8	Tiago Cezar	M	25	Araranguá
2	Gabriel Matias	M	22	Araranguá
6	Gustavo Pereira	M	18	Orleans
3	Leonardo Cunha	M	33	Cocal do Sul
9	Rafael Monteiro	M	29	Criciúma
5	Fernada Ricardo	F	35	Cocal do Sul
1	Leandro Fontanela	M	29	Urussanga
0	Maria da Silva	F	25	Criciúma
```

**x\_test com estratificação:**

```
	Nome	Sexo	Idade	Cidade
7	Leandro Melo	M	31	Criciúma
4	Ana Carolina Souza	F	25	Treviso
```

-----

##Método de validação K-Fold

K-fold é o método de cross-validation mais conhecido e utilizado. O método consiste em dividir o dataset em k partes, usando k-1 partes para treino e a parte remanescente para teste, fazendo isso k vezes. [cite: 62]

*Diagrama ilustrando o processo K-Fold com 5 iterações (K-Folds). Em cada iteração, um "fold" diferente (azul) é usado para validação, enquanto os outros quatro (cinza) são usados para treinamento. O desempenho de cada iteração é medido, e o desempenho final é a média dos desempenhos das 5 iterações.*

-----

## Praticando o método K-Fold

### Importando e criando o modelo K-Fold

```python
# Importando o método K-fold
from sklearn.model_selection import KFold

# Criando o modelo, definindo o número de pastas.
modelKFold = KFold(5, random_state=22, shuffle=True)

# Apresentando o número de divisões.
modelKFold.get_n_splits(dataframe)
```

**Output:**

```
5
```

### Iterando sobre as combinações

```python
# variável de controle
i = 1
# Percorrendo cada uma das combinações de pastas
for train_index, test_index in modelKFold.split(x):
	# Separando os dados em treino e teste para X e para Y
	x_train, x_test = x.loc[train_index], x.loc[test_index]
	y_train, y_test = y.loc[train_index], y.loc[test_index]
	# Printando as informações.
	print('------------Dados para Treino K-fold {} ---------------'.format(i))
	print('---------- X train ----------')
	print(x_train)
	print('---------- Y train ----------')
	print(y_train)
	print('---------- X test -----------')
	print(x_test)
	print('---------- Y test -----------')
	print(y_test)
	print('-------------------------------------------------------')
	i += 1
```

### Resultados das Iterações

**K-fold 1**

  * **Dados de Treino:** 8 instâncias (índices 0, 3, 4, 5, 6, 7, 8, 9)
  * **Dados de Teste:** 2 instâncias (índices 1, 2)
  * *Ações:* Treinar modelo 1, Avaliar Modelo 1, Armazenar métricas

**K-fold 2**

  * **Dados de Treino:** 8 instâncias (índices 0, 1, 2, 3, 4, 5, 7, 8)
  * **Dados de Teste:** 2 instâncias (índices 6, 9)
  * *Ações:* Treinar modelo 2, Avaliar Modelo 2, Armazenar métricas

**K-fold 3**

  * **Dados de Treino:** 8 instâncias (índices 0, 1, 2, 4, 5, 6, 8, 9)
  * **Dados de Teste:** 2 instâncias (índices 3, 7)
  * *Ações:* Treinar modelo 3, Avaliar Modelo 3 , Armazenar métricas

**K-fold 4**

  * **Dados de Treino:** 8 instâncias (índices 1, 2, 3, 4, 5, 6, 7, 9)
  * **Dados de Teste:** 2 instâncias (índices 0, 8)
  * *Ações:* Treinar modelo 4, Avaliar Modelo 4 , Armazenar métricas

[cite\_start]**K-fold 5** [cite: 87]

  * **Dados de Treino:** 8 instâncias (índices 0, 1, 2, 3, 6, 7, 8, 9)
  * **Dados de Teste:** 2 instâncias (índices 4, 5)
  * *Ações:* Treinar modelo 5, Avaliar Modelo 5, Armazenar métricas

### Performance Final

A performance final é a média das métricas de todos os modelos treinados.
$$\text{Performance Final} = \frac{\sum (\text{Métricas Modelo 1} + \text{...} + \text{Métricas Modelo 5})}{\text{Número de Pastas (K do K-fold)}}$$

-----

## Método de validação Leave One Out

É um caso específico do k-fold, com k igual ao número total de dados N. Nesta abordagem são realizados N cálculos de erro, um para cada dado.

### Performance Final

A performance final é a média das métricas dos 10 modelos (pois há 10 instâncias).
$$\text{Performance Final} = \frac{\sum (\text{Métricas Modelo 1} + \text{...} + \text{Métricas Modelo 10})}{\text{Número de instâncias}}$$

-----

## Método de validação Leave P Out

*(Esta seção descreve o método Leave P Out, uma variação onde `p` amostras são deixadas de fora para teste.)*

-----

## Método de validação Time Series

Fornece índices de treinamento/teste para dividir amostras de dados de série temporal que são observadas em intervalos de tempo fixos, em conjuntos de treinamento/teste. Em cada divisão, os índices de teste devem ser maiores do que antes e, portanto, embaralhar no validador cruzado é inadequado.
