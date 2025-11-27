1. Etapa3_Preparacao_Dados.ipynb
   - Notebook de preparação de dados
   - para Google Colab
   - Gera 7 arquivos CSV + 1 arquivo PKL

2. Etapa4_Analise_Exploratoria.ipynb
   - Notebook de análise exploratória
   - para Google Colab
   - Insights e recomendações

3. loan_approval_dataset.csv (data/)
   - Dataset original com 4.500 registros
   - 13 variáveis (12 features + 1 target)

COMO USAR


PASSO 1: Extrair o arquivo ZIP
   - Descompacte loan_approval_etapas3e4.zip

PASSO 2: Executar Etapa 3
   - Abra Etapa3_Preparacao_Dados.ipynb no Google Colab
   - Faça upload de loan_approval_dataset.csv
   - Execute: Runtime > Run all
   - Baixe os arquivos gerados (7 CSV + 1 PKL)

PASSO 3: Executar Etapa 4
   - Abra Etapa4_Analise_Exploratoria.ipynb no Google Colab
   - Faça upload de X_train.csv e y_train.csv (da Etapa 3)
   - Execute: Runtime > Run all
   - Analise os gráficos e insights

PASSO 4: Preparar Apresentação
   - Use os gráficos gerados na Etapa 4
   - Siga as sugestões de slides do GUIA_RAPIDO.md


ARQUIVOS GERADOS PELA ETAPA 3

Após executar Etapa3_Preparacao_Dados.ipynb, você terá:

1. X_train.csv - Features de treino (não normalizadas)
2. X_test.csv - Features de teste (não normalizadas)
3. X_train_scaled.csv - Features de treino (normalizadas)
4. X_test_scaled.csv - Features de teste (normalizadas)
5. y_train.csv - Target de treino
6. y_test.csv - Target de teste
7. scaler.pkl - Objeto StandardScaler (para deploy)

Total: 3.600 registros treino + 900 registros teste
Features: 17 variáveis (11 originais + 6 engineered)

