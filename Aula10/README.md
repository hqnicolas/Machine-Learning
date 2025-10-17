# K-Means Clustering

## K-Means Clustering

O K-means clustering é um dos algoritmos de aprendizado de máquina não supervisionado mais simples e populares. Normalmente, os algoritmos não supervisionados fazem inferências de conjuntos de dados usando apenas vetores de entrada, sem se referir a resultados conhecidos ou rotulados.

> “O objetivo do K-means é simples: agrupar pontos de dados semelhantes e descobrir padrões. Para atingir este objetivo, K-means procura um número fixo (*k*) de clusters em um conjunto de dados.”

### Diferença entre Classificação e Clustering

### Exemplos de Agrupamento Visual

O K-Means é poderoso para agrupar dados visualmente semelhantes.

*Cachorros ou Frango Frito?*

*Shiba Inu ou Marshmallows?*

-----

## Como o Algoritmo K-Means Funciona

O agrupamento k-Means consiste em colocar os pontos de treinamento que temos em agrupamentos. Queremos saber quais pontos de dados pertencem um ao outro, sem ter rótulos para nenhum deles.

### Pseudo-Código

O algoritmo pode ser resumido nos seguintes passos:

**Algoritmo 1: K-Means**

  * **Entrada:** base de dados com *i* instâncias e *d* dimensões, *K* grupos desejados
  * **Saída:** Base de dados dividida em *K* grupos

<!-- end list -->

1.  **início**
2.  inicializa os K centroides com valores aleatórios;
3.  **enquanto** critério de parada não atingido;
4.      **para cada** amostra xᵢ **faça**
5.          Adicione xᵢ ao grupo do centroide Cₖ de menor distância, de acordo com a equação (1);
6.      **fim**
7.      Atualiza os centroides Cₖ de acordo com a equação (2);
8.  **fim**

#### Fórmulas

1.  **Distância Euclidiana:** Usada para medir a distância entre um ponto e um centroide.

$$
d(x\_i, x\_j) = \\sqrt{\\sum\_{p=1}^{d} (x\_{i,p} - x\_{j,p})^2}
$$
    
3.  **Recálculo do Centroide:** Usado para atualizar a posição do centroide para a média de todos os pontos no seu cluster.

$$
C\_k = \\frac{1}{n\_k} \\sum\_{i=1}^{n\_k} x\_i^k
$$
    
### Passo a Passo do Algoritmo

1.  **Início**: Temos um conjunto de dados que queremos agrupar.

2.  **Inicialização**: O algoritmo começa colocando *k* centroides (neste caso, 3: A, B, C) em locais aleatórios.

3.  **Atribuição de Cluster**: O algoritmo mede a distância de cada ponto de dado para cada centroide e atribui o ponto ao cluster do centroide mais próximo.

4.  **Recálculo do Centroide**: Após todos os pontos serem atribuídos, o algoritmo recalcula a posição de cada centroide. O novo valor do centroide é a média de todos os pontos de dados pertencentes àquele cluster.

5.  **Repetição**: O processo de atribuição e recálculo é repetido até que a posição dos centroides não mude mais. Isso indica que os clusters estão estáveis. O resultado final é um conjunto de clusters, com cada centroide no centro de seu respectivo grupo.

### Animação do Processo

A animação abaixo demonstra as iterações do K-Means e a convergência dos clusters.

-----

## Exemplo Prático com Scikit-Learn: Sistema de Recomendação de Filmes

Vamos usar o K-Means para criar um sistema de recomendação de filmes simples.

### 1\. Importar Bibliotecas e Carregar Dados

Primeiro, importamos as bibliotecas necessárias e carregamos os datasets de filmes e avaliações.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Carregar os dados
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
```

### 2\. Análise Exploratória

Visualizamos as primeiras linhas dos datasets.

```python
# Exibir as primeiras linhas dos datasets
movies.head()
```

```python
ratings.head()
```

### 3\. Pré-processamento

Criamos uma matriz de interação usuário-filme, onde as linhas são usuários, as colunas são filmes e os valores são as avaliações. Preenchemos os valores `NaN` com 0.

```python
# Criar a matriz de interação
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Preencher valores NaN (usuários que não avaliaram certos filmes)
user_movie_matrix.fillna(0, inplace=True)
user_movie_matrix.head()
```

### 4\. Método do Cotovelo (Elbow Method)

Para encontrar o número ideal de clusters (*k*), usamos o Método do Cotovelo. Ele calcula a soma dos quadrados intra-cluster (WCSS) para diferentes valores de *k*.

```python
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(user_movie_matrix)
    wcss.append(kmeans.inertia_)

# Plotar o gráfico do cotovelo
plt.plot(range(1, 20), wcss)
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()
```

O "cotovelo" no gráfico sugere um bom número de clusters. Vamos escolher 7 para este exemplo.

### 5\. Aplicar o K-Means e Visualizar os Clusters

Agora, aplicamos o K-Means com *k*=7 e visualizamos a distribuição dos usuários por cluster.

```python
# Aplicar K-means com 7 clusters
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=42)
user_clusters = kmeans.fit_predict(user_movie_matrix)

# Visualizar a distribuição
plt.figure(figsize=(8,6))
sns.countplot(x=user_clusters, palette='viridis')
plt.title('Distribuição de Usuários por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de Usuários')
plt.show()
```

### 6\. Gerar Recomendações

Para um usuário específico (ex: `userId = 500`), podemos encontrar seu cluster e recomendar filmes que outros usuários nesse mesmo cluster avaliaram bem.

```python
# Adicionar o cluster de cada usuário ao dataset original
user_movie_matrix['cluster'] = user_clusters

# Escolher um usuário (por exemplo, userId = 500) e encontrar o cluster dele
user_id = 500
user_cluster = user_movie_matrix.loc[user_id, 'cluster']

# Filtrar todos os usuários no mesmo cluster
cluster_users = user_movie_matrix[user_movie_matrix['cluster'] == user_cluster]

# Calcular a média de avaliação dos filmes pelos usuários no cluster
cluster_mean_ratings = cluster_users.drop('cluster', axis=1).mean()

# Ordenar os filmes com base nas avaliações médias e recomendar os top 5
recommended_movies = cluster_mean_ratings.sort_values(ascending=False).head(5)

# Exibir os nomes dos filmes recomendados
recommended_movie_ids = recommended_movies.index
recommended_movie_titles = movies[movies['movieId'].isin(recommended_movie_ids)]['title']

print(f"Recomendações de filmes para o usuário {user_id}:")
print(recommended_movie_titles)
```
