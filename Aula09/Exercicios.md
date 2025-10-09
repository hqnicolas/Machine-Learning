Vamos calcular as métricas de avaliação do modelo de classificação (Acurácia, Precisão, Recall e F1-Score), usando os dados fornecidos.

### ✅ Dados fornecidos:

| | **Classe Real: Não Sobreviveu (Negativa)** | **Classe Real: Sobreviveu (Positiva)** |
| :--- | :--- | :--- |
| **Previsto: Não Sobreviveu** | 493 (Verdadeiro Negativo - TN) | 56 (Falso Negativo - FN) |
| **Previsto: Sobreviveu** | 97 (Falso Positivo - FP) | 245 (Verdadeiro Positivo - TP) |

---

### 📌 1. **Acurácia**
É a proporção de previsões corretas (positivas e negativas) sobre o total de casos.

$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Acurácia} = \frac{245 + 493}{245 + 493 + 97 + 56} = \frac{738}{891} \approx 0.828 \text{ ou } 82.8\%$$

---

### 📌 2. **Precisão (Precision)**
É a proporção de verdadeiros positivos sobre o total de previstos como positivos.

$$\text{Precisão} = \frac{TP}{TP + FP}$$

$$\text{Precisão} = \frac{245}{245 + 97} = \frac{245}{342} \approx 0.716 \text{ ou } 71.6\%$$

---

### 📌 3. **Recall (Sensibilidade)**
É a proporção de verdadeiros positivos sobre o total de positivos reais.

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Recall} = \frac{245}{245 + 56} = \frac{245}{301} \approx 0.814 \text{ ou } 81.4\%$$

---

### 📌 4. **F1-Score**
É a média harmônica entre precisão e recall.

$$\text{F1-Score} = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

$$\text{F1-Score} = 2 \cdot \frac{0.716 \cdot 0.814}{0.716 + 0.814} = 2 \cdot \frac{0.582}{1.53} \approx 0.761 \text{ ou } 76.1\%$$

---

### ✅ Resumo Final:

| Métrica | Valor Aproximado |
| :--- | :--- |
| Acurácia | **82.8%** |
| Precisão | **71.6%** |
| Recall | **81.4%** |
| F1-Score | **76.1%** |
