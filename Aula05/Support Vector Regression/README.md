# 📈 Support Vector Regression (SVR)

Uma visão geral sobre o funcionamento, implementação e aplicação do SVR, um poderoso algoritmo de regressão baseado em Support Vector Machines (SVM).

-----

## 📖 Introdução

### O que é SVR?

**Support Vector Regression (SVR)** é uma adaptação do popular algoritmo de classificação *Support Vector Machine (SVM)* para resolver problemas de regressão. Enquanto o SVM busca encontrar um hiperplano que melhor separe os dados em classes, o SVR busca encontrar um hiperplano que melhor se ajuste aos dados contínuos.

### Origem

O SVR, assim como o SVM, foi desenvolvido nos Laboratórios da AT\&T por Vladimir Vapnik e seus colegas, consolidando-se como uma técnica robusta e eficaz de aprendizado de máquina na década de 1990.

### 📏 O Tubo Epsilon-Insensitive (`ε-Insensitive Tube`)

A principal inovação do SVR é o conceito do **tubo epsilon-insensitive**. Diferente de outras regressões que tentam minimizar o erro para *todos* os pontos, o SVR define uma margem de tolerância, chamada de **epsilon ($\\epsilon$)**.

O objetivo do algoritmo não é minimizar o erro, mas sim garantir que o maior número possível de pontos de dados esteja **dentro** desse tubo, onde o erro é considerado zero.

### Termos Essenciais

  - **Margem de Tolerância ($\\epsilon$):** É a largura do tubo. Pontos de dados que caem dentro desta margem não recebem nenhuma penalidade.
  - **Support Vectors (Vetores de Suporte):** São os pontos de dados que estão na borda ou fora do tubo epsilon. São eles que "suportam" e definem o hiperplano de regressão.
  - **Termo de Regularização (C):** Um hiperparâmetro que controla o trade-off entre a "planicidade" (simplicidade) do modelo e a quantidade de pontos que podem ficar fora da margem.
      - **`C` baixo:** Modelo mais simples, margem maior, mais erros são tolerados.
      - **`C` alto:** Modelo mais complexo, margem menor, tenta se ajustar melhor aos dados de treino (risco de overfitting).
  - **Kernel Trick:** Uma técnica matemática que permite ao SVR modelar relações não-lineares. Os kernels mais comuns são:
      - **Linear:** Para problemas linearmente separáveis.
      - **Polinomial:** Para relações polinomiais.
      - **RBF (Radial Basis Function):** O mais popular e flexível, capaz de lidar com relações complexas.

-----

## ⚙️ Funcionamento do Algoritmo

O SVR busca resolver um problema de otimização com dois objetivos principais:

1.  **Maximizar a "planicidade" (flatness) da função:** O modelo tenta encontrar a função mais simples (menos "curvada") possível que se ajuste aos dados.
2.  **Minimizar a penalização dos pontos fora do tubo:** Para cada ponto fora da margem $\\epsilon$, uma penalidade é aplicada. A soma dessas penalidades deve ser minimizada.

Para resolver esse problema de otimização de forma eficiente, são utilizadas abordagens como a **Sequential Minimal Optimization (SMO)**, que divide o problema complexo em subproblemas menores, reduzindo drasticamente o custo computacional.

-----

## 💻 Implementação Prática (Scikit-Learn)

A biblioteca `scikit-learn` oferece uma implementação simples e poderosa do SVR. O exemplo abaixo mostra como treinar um modelo básico.

```python
# Importando as bibliotecas necessárias
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Gerando dados de exemplo
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

# Criando e treinando o modelo SVR
# 1. StandardScaler: Normaliza os dados (passo importante para o SVR)
# 2. SVR: Define o modelo com os hiperparâmetros C e epsilon
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Treinando o modelo com os dados
regr.fit(X, y)

# Agora o modelo 'regr' está pronto para fazer previsões
# regr.predict(novos_dados)
```

-----

## 🌐 Aplicações Práticas

O SVR é especialmente útil em cenários onde a relação entre as variáveis não é linear e o número de features é alto. Algumas aplicações incluem:

  - **Previsão de Séries Temporais:** Como prever o preço de ações ou a demanda de um produto.
  - **Projetos de Energia:** Estimar a geração de energia de painéis solares com base em dados climáticos.
  - **Geociências:** Modelagem de parâmetros geofísicos.
  - **Finanças:** Avaliação de risco de crédito e previsão de volatilidade.

Em comparação com a **Regressão Linear**, o SVR tende a ser mais robusto a outliers e mais flexível para capturar padrões complexos, graças ao `kernel trick`.

-----

## ❓ Perguntas e Respostas

#### 1\. O que o modelo faz com os pontos que ficam fora do tubo?

Os pontos fora do tubo são aqueles que o modelo errou por uma margem maior que o $\\epsilon$ (epsilon) definido. O algoritmo aplica uma **penalidade** a esses pontos, que é proporcional à distância deles até a borda do tubo.

O objetivo do treinamento é encontrar um equilíbrio: criar um tubo que contenha o máximo de pontos possível, minimizando ao mesmo tempo a soma total das penalidades dos pontos que ficaram de fora. Esses pontos penalizados são cruciais, pois eles se tornam os **Vetores de Suporte** que definem a fronteira da regressão.

#### 2\. Suponha que você tenha um tubo muito estreito. Como isso afeta a previsão para os pontos que estão fora da faixa central?

Um tubo muito estreito (um valor de `epsilon` muito pequeno) significa que o modelo tem uma **tolerância muito baixa a erros**. Consequentemente:

  - **O modelo se torna mais complexo:** Para tentar encaixar mais pontos dentro ou perto do tubo estreito, a função de regressão se tornará mais "curvada" e complexa, seguindo os dados de treino mais de perto.
  - **Aumenta o risco de overfitting:** O modelo pode começar a aprender o ruído dos dados de treinamento em vez do padrão geral.
  - **Maior sensibilidade a outliers:** Pontos que estão fora do tubo terão uma influência maior na definição do modelo, pois o algoritmo se esforçará mais para reduzir o erro associado a eles.

Em resumo, um tubo estreito força o modelo a ser muito preciso nos dados de treino, o que pode prejudicar sua capacidade de generalizar e fazer boas previsões para novos dados.
