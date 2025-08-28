## 👩‍💻 Aula 4: Regressão Linear e KNN

### O que é Regressão Linear?

Modelos de **regressão** são modelos matemáticos que estabelecem uma relação entre uma **variável dependente (Y)** e uma ou mais **variáveis independentes (X)**. Em outras palavras, eles tentam prever o valor de Y com base no valor de X.

* **Variável Independente (X)** ou **Variável Preditiva**: É a variável de entrada que o modelo usa para fazer a previsão. Ela influencia a variável que queremos encontrar.
* **Variável Dependente (Y)** ou **Variável Alvo**: É a variável que queremos prever.

---

### Tipos de Regressão Linear

Existem dois tipos principais de regressão linear:

* **Regressão Linear Simples**: Usa apenas uma **única variável independente (X)** para fazer a predição.
* **Regressão Linear Múltipla**: Usa **várias variáveis independentes (X)** para fazer a predição.

---

### A Equação do Modelo de Regressão Linear

O modelo de regressão linear pode ser representado pela seguinte equação:

$Y = \alpha + \beta X + \epsilon$

* **Y**: Variável **dependente** (o que o modelo tentará prever).
* **$\alpha$**: A **constante** (ou intercepto), representa onde a linha de regressão cruza o eixo Y.
* **$\beta$**: O **coeficiente** (ou inclinação), representa a inclinação da linha em relação à variável explicativa.
* **X**: Variável **explicativa** (independente).
* **$\epsilon$**: Os **valores residuais**, que representam possíveis erros ou a parte da variável Y que não pode ser explicada por X.

---

### Método de Separação: Treino e Teste (Holdout)

O método **Holdout** é uma técnica comum para avaliar a performance de um modelo. Ele consiste em dividir o conjunto total de dados em dois subconjuntos:

* **Conjunto de Treino**: Usado para **treinar** o modelo e estimar seus parâmetros. Geralmente corresponde a cerca de **70%** dos dados.
* **Conjunto de Teste**: Usado para **avaliar a performance** do modelo com dados que ele nunca viu. Geralmente corresponde a cerca de **30%** dos dados restantes.

Essa separação ajuda a garantir que a avaliação do modelo seja feita de forma justa, evitando que ele "decore" os dados de treino e tenha uma performance ruim em novos dados.
