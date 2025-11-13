## O Que é Deploy de Modelos de ML?

Um modelo de ML só traz valor quando está em produção, sendo usado para tomar decisões ou automações.

Deploy é o processo de disponibilizar um modelo treinado para uso em um ambiente real. Fazer com que o modelo gere previsões para novos dados automaticamente.

### Ciclo de Vida de ML

1.  Coleta de Dados
2.  Pré-processamento
3.  Treinamento
4.  Avaliação
5.  **Deploy**
6.  Monitoramento e Re-treino

---

## Tipos de Deploy de Modelos

### Batch
* O modelo processa dados periodicamente (ex.: previsões diárias).
* **Exemplo:** Previsão de demanda semanal.
  
<img width="1624" height="464" alt="image" src="https://github.com/user-attachments/assets/e3cf000c-bd04-43bc-af02-834bf0d3ad81" />


### Online/Real-Time
* O modelo recebe e processa dados em tempo real.
* **Exemplo:** Previsões de recomendação em um e-commerce.

<img width="1548" height="401" alt="image" src="https://github.com/user-attachments/assets/6398f0df-cd17-4f9c-a8e1-2f7df2a16fc8" />


### Edge Deployment
* Modelos rodando em dispositivos locais com recursos limitados.
* **Exemplo:** Reconhecimento de imagens em dispositivos móveis.

<img width="1488" height="477" alt="image" src="https://github.com/user-attachments/assets/71d8c5a2-58ef-4a05-a00a-0294c7b5a982" />


*(Diagrama de TheAiEdge.io ilustrando os tipos de deploy: Batch, Real-time, Streaming, Edge)*

---

## Infraestrutura para Deploy

### On-Premise
* **Definição:** Servidores locais, mais controle e customização.
* **Vantagens:** Controle completo e segurança.
* **Desvantagens:** Custo elevado e manutenção complexa.

### Cloud (AWS, GCP, Azure)
* **Definição:** Provedores de nuvem oferecem escalabilidade e ferramentas prontas.
* **Vantagens:** Flexibilidade, custo por uso, escalabilidade automática.
* **Desvantagens:** Dependência de terceiros, custos variáveis.

---

## Principais Desafios de um Deploy

* **Integração:** Fazer com que o modelo se comunique com sistemas legados ou APIs.
* **Escalabilidade:** Garantir que o sistema continue funcionando à medida que o volume de dados aumenta.
* **Latência:** Reduzir o tempo entre a solicitação de uma previsão e a resposta.
* **Manutenção:** Monitorar a performance do modelo para garantir sua efetividade ao longo do tempo.

---

## Utilizando Pickle para Salvar e Carregar um Modelo de ML

```python
# Importa as bibliotecas necessárias
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

# Define a URL para o dataset Pima Indians Diabetes que será utilizado
url = "[https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)"

# Define os nomes das colunas para o dataframe
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Lê o dataset a partir da URL e cria um dataframe
dataframe = pandas.read_csv(url, names=names)

# Converte o dataframe em um array de valores
array = dataframe.values

# Separa os dados de entrada (features) e as saídas (rótulos)
X = array[:,0:8] # Features (colunas de 0 a 7)
Y = array[:,8] # Rótulo (coluna 8)

# Define o tamanho do conjunto de teste (33% dos dados) e uma semente para replicação dos resultados
test_size = 0.33
seed = 7

# Divide o conjunto de dados em treinamento e teste, de acordo com o tamanho e a semente definidos
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,
                                                                    Y,
                                                                    test_size=test_size,
                                                                    random_state=seed)

# Cria um modelo de Regressão Logística
model = LogisticRegression()

# Ajusta o modelo aos dados de treinamento
model.fit(X_train, Y_train)

# Salva o modelo treinado em disco usando o Pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Em um momento posterior...

# Carrega o modelo salvo do disco
loaded_model = pickle.load(open(filename, 'rb'))

# Exemplo de entrada
input_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

# Avalia o modelo carregado e realiza predição
result = loaded_model.predict(input_data)
print(result)
````

-----

## Utilizando Joblib para Salvar e Carregar um Modelo de ML

```python
# Importa a biblioteca joblib para salvar e carregar o modelo
import joblib

# Salva o modelo treinado em disco usando joblib
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# Em um momento posterior...

# Carrega o modelo salvo do disco usando joblib
loaded_model = joblib.load(filename)

# Exemplo de entrada
input_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

# Realiza a predição do exemplo da entrada
result = loaded_model.predict(input_data)
print(result)
```

-----

## Deploy com Flask e FastAPI

### Flask

  * **Deploying Machine Learning Models with Flask: A Step-by-Step Guide** (Talent500)
      * Mostra como treinar/salvar um modelo, depois carregá-lo no Flask e criar endpoint para predição.
  * **How to Deploy Machine Learning Models using Flask (with Code\!)** (Analytics Vidhya)
      * Tutorial mais antigo, mas bom para entender os conceitos básicos.
  * **Turning Machine Learning Models into APIs in Python** (via DataCamp)
      * Foca em como transformar um modelo em API usando Flask.

### FastAPI

  * **Deploying Machine Learning models using FastAPI (Medium)** (Medium)
      * Mostra passo a passo com FastAPI, Pydantic, carregamento de modelo e endpoint de predição.
  * **How to Use FastAPI for Machine Learning** (The JetBrains Blog)
      * Aborda o framework FastAPI para ciência de dados, inclusive uso para servir modelos.
  * **Step-by-Step Guide to Deploying Machine Learning Models with FastAPI and Docker** (MachineLearningMastery.com)
      * Mais completo, incluindo Docker além de FastAPI.
  * **Integrating Machine Learning Models with FastAPI** (App Generator)
      * Guia prático focado em carregar modelo, criar app, endpoint, etc.

