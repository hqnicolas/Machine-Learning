# Lista de Exercícios

## Parte 1: Avaliação de Modelos de Classificação

Para cada um dos cenários abaixo, calcule as métricas de avaliação do modelo (Acurácia, Precisão, Recall e F1-Score) e monte a Matriz de Confusão.

---

### Exercício 1: Detecção de Fraudes Financeiras

Um modelo de detecção de fraudes em transações financeiras obteve os seguintes resultados:

* **Verdadeiros Negativos (VN):** 450 transações foram corretamente classificadas como não fraudulentas.
* **Falsos Negativos (FN):** 30 transações foram erroneamente classificadas como não fraudulentas (eram fraudes).
* **Falsos Positivos (FP):** 20 transações foram erroneamente classificadas como fraudulentas (não eram fraudes).
* **Verdadeiros Positivos (VP):** 100 transações foram corretamente classificadas como fraudulentas.

---

### Exercício 2: Diagnóstico Médico

Um sistema de diagnóstico médico para detecção de uma doença obteve os seguintes resultados:

* **Verdadeiros Negativos (VN):** 800 pacientes foram corretamente diagnosticados como saudáveis.
* **Falsos Negativos (FN):** 20 pacientes foram erroneamente diagnosticados como saudáveis (estavam doentes).
* **Falsos Positivos (FP):** 30 pacientes foram erroneamente diagnosticados como doentes (estavam saudáveis).
* **Verdadeiros Positivos (VP):** 150 pacientes foram corretamente diagnosticados como doentes.

---

### Exercício 3: Detecção de Spam

Um modelo de detecção de spam em mensagens de texto obteve os seguintes resultados:

* **Verdadeiros Negativos (VN):** 1200 mensagens foram corretamente classificadas como não spam.
* **Falsos Negativos (FN):** 40 mensagens foram erroneamente classificadas como não spam (eram spam).
* **Falsos Positivos (FP):** 60 mensagens foram erroneamente classificadas como spam (não eram spam).
* **Verdadeiros Positivos (VP):** 200 mensagens foram corretamente classificadas como spam.

---

### Exercício 4: Reconhecimento Facial

Um sistema de reconhecimento facial para identificar a presença de uma pessoa em imagens obteve os seguintes resultados:

* **Verdadeiros Positivos (VP):** 1500 imagens de pessoas foram corretamente identificadas.
* **Falsos Positivos (FP):** 50 imagens sem pessoas foram erroneamente identificadas como tendo pessoas.
* **Falsos Negativos (FN):** 30 imagens de pessoas foram erroneamente identificadas como não tendo pessoas.
* **Verdadeiros Negativos (VN):** 300 imagens sem pessoas foram corretamente identificadas como não tendo pessoas.

---

## Parte 2: Projetos Práticos de Machine Learning

### Exercício 5: Projeto de Classificação

Você foi encarregado de resolver um problema de classificação utilizando aprendizado de máquina. Para isso, siga os passos abaixo:

1.  **Seleção do Dataset:** Escolha um dataset adequado para uma tarefa de classificação. O dataset pode ser obtido de repositórios públicos, como o [Kaggle](https://www.kaggle.com/datasets) ou [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
2.  **Divisão dos Dados:** Divida o dataset em dois conjuntos: um de **treino** e outro de **teste**, garantindo que o conjunto de teste represente uma amostra significativa dos dados.
3.  **Treinamento do Modelo:** Treine um modelo de aprendizado de máquina de sua escolha (por exemplo, uma Árvore de Decisão, Regressão Logística, SVM, ou qualquer outro classificador adequado) utilizando o conjunto de treino.
4.  **Previsão:** Após o treinamento, aplique o modelo ao conjunto de teste e obtenha as previsões.
5.  **Avaliação:** Avalie o desempenho do modelo utilizando a **Matriz de Confusão** e calcule métricas como **Acurácia**, **Precisão**, **Recall** e **F1-Score** com base nos resultados do conjunto de teste.
6.  **Análise dos Resultados:** Comente sobre a performance do modelo com base nas métricas extraídas.

---

### Exercício 6: Projeto de Regressão

Você foi encarregado de resolver um problema de regressão utilizando aprendizado de máquina. Para isso, siga os passos abaixo:

1.  **Seleção do Dataset:** Escolha um dataset adequado para uma tarefa de regressão de repositórios públicos.
2.  **Divisão dos Dados:** Divida o dataset em conjuntos de **treino** e **teste**.
3.  **Treinamento do Modelo:** Treine um modelo de aprendizado de máquina de sua escolha (por exemplo, Regressão Linear, Árvore de Decisão para Regressão, ou outro modelo adequado) utilizando o conjunto de treino.
4.  **Previsão:** Após o treinamento, aplique o modelo ao conjunto de teste para realizar as previsões.
5.  **Avaliação:** Avalie o desempenho do modelo utilizando métricas de regressão, como o **Erro Quadrático Médio (MSE)**, **Erro Absoluto Médio (MAE)** e **R² (Coeficiente de Determinação)**, entre outras que considerar pertinentes.
6.  **Análise dos Resultados:** Comente sobre a performance do modelo com base nas métricas extraídas e identifique possíveis ajustes ou melhorias no modelo.

---

### Exercício 7: Persistência de Modelos

Neste exercício, você irá trabalhar com a persistência de modelos de aprendizado de máquina, utilizando as bibliotecas `Pickle` e `Joblib` para salvar e carregar os modelos treinados. Utilize os modelos de classificação e regressão criados nos exercícios anteriores e siga as instruções abaixo:

1.  **Seleção dos Modelos:** Utilize os modelos de classificação e regressão que você treinou nos exercícios 5 e 6.
2.  **Salvar os Modelos:**
    * Após treinar cada modelo, utilize `Pickle` ou `Joblib` para salvar os modelos treinados em arquivos no disco (ex: `modelo_classificacao.pkl`, `modelo_regressao.joblib`).
3.  **Carregar os Modelos:**
    * Após salvar, simule uma nova sessão de trabalho (ou em um novo script/célula) e carregue os modelos a partir dos arquivos salvos.
4.  **Realizar Previsões:** Utilize os modelos carregados para realizar novas previsões nos seus respectivos conjuntos de teste e verifique se os resultados são os mesmos obtidos antes de salvar.
