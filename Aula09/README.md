# Métricas de Avaliação em Machine Learning

## Introdução

Durante o processo de criação de um modelo de machine learning, é fundamental medir sua qualidade de acordo com o objetivo da tarefa. Existem funções matemáticas que nos ajudam a avaliar a capacidade de erro e acerto dos nossos modelos. Tão importante quanto saber escolher um bom modelo, é saber escolher a métrica correta para decidir qual é o melhor entre eles.

-----

## Métricas para Problemas de Classificação

As métricas mais comuns para problemas de classificação são:

  * Acurácia
  * Precisão
  * Recall (Sensibilidade)
  * Especificidade
  * F1-Score

Essas métricas geralmente são calculadas a partir de uma **Matriz de Confusão**.

### Matriz de Confusão

A matriz de confusão é uma tabela que resume o desempenho de um modelo de classificação, comparando os valores previstos com os valores reais. Ela indica os erros e acertos do modelo.

A estrutura da matriz é a seguinte:

| | **Detectado: Sim** | **Detectado: Não** |
| :--- | :--- | :--- |
| **Real: Sim** | Verdadeiro Positivo (VP) | Falso Negativo (FN) |
| **Real: Não** | Falso Positivo (FP) | Verdadeiro Negativo (VN) |

  * **Verdadeiro Positivo (VP):** O modelo previu "Sim" e o valor real era "Sim".
  * **Verdadeiro Negativo (VN):** O modelo previu "Não" e o valor real era "Não".
  * **Falso Positivo (FP):** O modelo previu "Sim", mas o valor real era "Não" (Erro Tipo I).
  * **Falso Negativo (FN):** O modelo previu "Não", mas o valor real era "Sim" (Erro Tipo II).

#### Exemplo Prático de Matriz de Confusão

Vamos considerar um dataset para prever se uma pessoa possui uma doença.

```python
# Importando as libs
import pandas as pd
import numpy as np

# Carregando csv.
dataframe = pd.read_csv('Dataset1.csv',encoding= 'unicode_escape',sep=';')
```

**Dataset:**
| | Nome | Valor real possuir a doença | Valor modelo possuir a doença |
|---|---|---|---|
| 0 | João | 0 | 1 |
| 1 | Maria | 0 | 1 |
| 2 | Pedro | 1 | 1 |
| 3 | Mateus | 0 | 1 |
| 4 | Rafael | 0 | 0 |
| 5 | Joana | 0 | 0 |
| 6 | Fernanda | 0 | 0 |
| 7 | Thiago | 1 | 0 |
| 8 | Gabriela | 0 | 0 |
| 9 | Daniel | 1 | 1 |

Calculando os valores da matriz de confusão com Scikit-learn:

```python
# Pegando os valores verdadeiros.
yTrue = dataframe['Valor real possuir a doença']

# Pegando os valores preditos pelo modelo.
yPredict = dataframe['Valor modelo possuir a doença']

# Importando a lib para extrair a matriz de confusão.
from sklearn.metrics import confusion_matrix

# Pegando os valores TN FP FN TP.
tn, fp, fn, tp = confusion_matrix(yTrue,yPredict).ravel()

print(f'Verdadeiro Negativo: {tn}')
print(f'Falso Positivo: {fp}')
print(f'Falso Negativo: {fn}')
print(f'Verdadeiro Positivo: {tp}\n')
```

**Resultado:**

  * Verdadeiro Negativo: 4
  * Falso Positivo: 3
  * Falso Negativo: 1
  * Verdadeiro Positivo: 2

Visualizando a Matriz de Confusão:

```python
import seaborn as sns

# Armazenando a matriz de confusão.
cfMatrix = confusion_matrix(yTrue,yPredict)

# Plotando a Matriz de Confusão
sns.heatmap(cfMatrix, annot=True)

# Plotando em percentual
sns.heatmap(cfMatrix/np.sum(cfMatrix), annot=True,
            fmt='.2%', cmap='Blues')
```

### 1\. Acurácia (Accuracy)

Indica a performance geral do modelo. É a razão entre o número de previsões corretas e o número total de exemplos.

**Quando usar?** É mais útil quando o dataset tem classes balanceadas (mesma proporção de exemplos para cada classe) e os custos de falsos positivos e falsos negativos são semelhantes.

**Fórmula:**
$$Acurácia = \frac{VP + VN}{VP + VN + FP + FN}$$

**Exemplo de Cálculo:**
Usando os valores do exemplo anterior:

  * VP = 2
  * VN = 4
  * Total de Observações = 10

$$Acurácia = \frac{2 + 4}{10} = 0,6 = 60\%$$

**Código:**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yTrue, yPredict)
print(f'{accuracy * 100}%') # 60.0%
```

**Cuidado:** A acurácia pode ser enganosa em datasets desbalanceados. Por exemplo, se 95% dos casos são negativos, um modelo que sempre prevê "negativo" terá 95% de acurácia, mas não será útil para identificar os casos positivos.

### 2\. Precisão (Precision)

Mede, dentre todas as previsões que o modelo classificou como positivas, quantas estavam realmente corretas. É uma boa métrica para quando o custo de um Falso Positivo é alto.

**Fórmula:**
$$Precisão = \frac{VP}{VP + FP}$$

**Exemplo de Cálculo:**

  * VP = 2
  * FP = 3

$$Precisão = \frac{2}{2 + 3} = 0,4 = 40\%$$

**Código:**

```python
from sklearn.metrics import precision_score

precision = precision_score(yTrue, yPredict)
print(f'{precision * 100}%') # 40.0%
```

### 3\. Recall (Sensibilidade)

Mede a capacidade do modelo de encontrar todos os casos positivos. Dentre todos os exemplos que são realmente positivos, quantos o modelo conseguiu identificar. É uma boa métrica quando o custo de um Falso Negativo é alto (por exemplo, em diagnósticos médicos).

**Fórmula:**
$$Recall = \frac{VP}{VP + FN}$$

**Exemplo de Cálculo:**

  * VP = 2
  * FN = 1

$$Recall = \frac{2}{2 + 1} \approx 0,66 = 66\%$$

**Código:**

```python
from sklearn.metrics import recall_score

recall = recall_score(yTrue, yPredict)
print(f'{recall * 100}%') # 66.66%
```

### 4\. F1-Score

É a média harmônica entre Precisão e Recall. É uma ótima métrica quando você precisa de um equilíbrio entre as duas, especialmente em casos de classes desbalanceadas.

**Fórmula:**
$$F1-Score = 2 \times \frac{Precisão \times Recall}{Precisão + Recall}$$

**Exemplo de Cálculo:**

  * Precisão = 0.40
  * Recall = 0.66

$$F1-Score = 2 \times \frac{0.40 \times 0.66}{0.40 + 0.66} \approx 0,50 = 50\%$$

**Código:**

```python
from sklearn.metrics import f1_score

f1 = f1_score(yTrue, yPredict)
print(f'{f1 * 100}%') # 50.0%
```

-----

## Métricas para Problemas de Regressão

Em problemas de regressão, o objetivo é prever um valor contínuo. As métricas avaliam o quão perto as previsões do modelo estão dos valores reais.

As métricas mais comuns são:

  * Mean Absolute Error (MAE) - Erro Absoluto Médio
  * Mean Squared Error (MSE) - Erro Quadrático Médio
  * Root Mean Squared Error (RMSE) - Raiz do Erro Quadrático Médio
  * R² Score - Coeficiente de Determinação

### 1\. Erro Médio Absoluto (Mean Absolute Error - MAE)

Calcula a média da diferença absoluta entre os valores previstos e os valores reais. É fácil de interpretar, pois está na mesma unidade da variável de saída.

**Fórmula:**
$$MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2\. Erro Médio Quadrático (Mean Squared Error - MSE)

Calcula a média dos erros ao quadrado. Ele penaliza mais os erros maiores, pois eleva as diferenças ao quadrado. Isso o torna sensível a outliers.

**Fórmula:**

$$
MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### 3\. Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE)

É a raiz quadrada do MSE. Assim como o MAE, o RMSE também está na mesma unidade da variável de saída, o que facilita a interpretação. Ele continua a penalizar erros maiores, mas o resultado é mais diretamente interpretável do que o MSE.

**Fórmula:**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 4\. R² Score (Coeficiente de Determinação)

Representa a proporção da variância da variável dependente que é explicada pelo modelo. Varia de 0 a 1 (ou 0% a 100%). Um R² de 1 indica que o modelo explica perfeitamente a variabilidade dos dados. Um R² de 0 indica que o modelo não explica nada. Um valor baixo sugere que o modelo de regressão não é válido.

**Fórmula:**

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
