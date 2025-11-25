
TRABALHO FINAL DE MACHINE LEARNING
ETAPAS 3 E 4 - ENTREGÁVEIS


Responsáveis: Nicolas Pereira & Nícolas M. Cardoso
Data: 25 de Novembro de 2025


ARQUIVOS PRINCIPAIS


1. Etapa3_Preparacao_Dados.ipynb
   - Notebook completo de preparação de dados
   - Pronto para Google Colab
   - Inclui todas as etapas de pré-processamento
   - Gera 7 arquivos CSV + 1 arquivo PKL

2. Etapa4_Analise_Exploratoria.ipynb
   - Notebook completo de análise exploratória
   - Pronto para Google Colab
   - 12+ visualizações profissionais
   - Insights detalhados e recomendações

3. loan_approval_dataset.csv (data/)
   - Dataset original com 4.500 registros
   - 13 variáveis (12 features + 1 target)
   - Pronto para uso nos notebooks


DOCUMENTAÇÃO


4. README.md
   - Guia completo do projeto
   - Instruções de uso dos notebooks
   - Estrutura do projeto
   - Checklist de entrega

5. GUIA_RAPIDO.md
   - Guia de execução rápida
   - Solução de problemas comuns
   - Dicas para apresentação
   - Próximos passos

6. RESUMO_EXECUTIVO.md
   - Resumo detalhado das etapas 3 e 4
   - Principais descobertas e insights
   - Recomendações para próximas etapas
   - Estatísticas finais

7. dataset_info.md
   - Informações sobre o dataset
   - Descrição das variáveis
   - Fonte e características


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
   - Consulte RESUMO_EXECUTIVO.md para insights principais
   - Siga as sugestões de slides do GUIA_RAPIDO.md


PRINCIPAIS INSIGHTS


1. DESBALANCEAMENTO DAS CLASSES
   - 66% Aprovados vs 34% Rejeitados
   - Acurácia NÃO é uma boa métrica
   - Foco em RECALL e AUC Score

2. FEATURE MAIS IMPORTANTE
   - credit_income_ratio (Relação Crédito/Renda)
   - Correlação -0.45 com aprovação
   - Criada via engenharia de features

3. PERFIL DE RISCO
   - Clientes rejeitados têm maior endividamento
   - CIBIL Score menor
   - Menor capacidade de pagamento

4. RECOMENDAÇÕES PARA MODELAGEM
   - Usar LightGBM ou XGBoost
   - Aplicar scale_pos_weight ou SMOTE
   - Focar em Recall e AUC Score
   - Considerar SHAP para interpretabilidade


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


PRÓXIMAS ETAPAS


ETAPA 5: Modelagem (Responsável: Pedro H. H. Marques)
   - Receber arquivos processados da Etapa 3
   - Implementar baseline (Regressão Logística)
   - Implementar modelo avançado (LightGBM/XGBoost)
   - Tratar desbalanceamento (scale_pos_weight)
   - Otimizar hiperparâmetros (GridSearchCV)
   - Avaliar com métricas adequadas (Recall, AUC)
   - Gerar SHAP plots para interpretabilidade

ETAPA 6: Deploy (Responsável: Gabriel M. Zavarize)
   - Receber modelo treinado (.pkl) do Pedro
   - Criar interface Streamlit/Gradio
   - Implementar mesma codificação e normalização
   - Testar predições

ETAPA 7: Apresentação (Responsável: Wilian V. Fernandes)
   - Consolidar notebooks
   - Preparar slides (10-15 minutos)
   - Incluir gráficos principais
   - Escrever conclusão



