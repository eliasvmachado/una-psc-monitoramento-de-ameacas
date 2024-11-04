# 💳 Monitoramento de Fraudes em Cartões de Crédito

## 🎓 Sobre o Projeto
Este é um projeto acadêmico desenvolvido para a disciplina de Sistemas Computacionais e Segurança. O objetivo é criar uma aplicação interativa para simular a detecção de fraudes em transações de cartões de crédito usando Machine Learning. O modelo utilizado foi baseado em um projeto no Kaggle, que alcançou uma precisão de 93% na identificação de fraudes. Você pode acessar o modelo original [aqui](https://www.kaggle.com/code/gallo33henrique/ml-creditcard-fraud-lightgbm-93-accuracy#-Machine-learning---Credit-Card-Fraud-Detection-).

Nesta aplicação, utilizamos a biblioteca Streamlit para oferecer uma interface visual e interativa que demonstra o funcionamento do monitoramento de fraudes em tempo real.

## 📊 Funcionalidades
- **Monitoramento em Tempo Real**: A aplicação permite simular a análise de transações, identificando e categorizando transações normais e fraudulentas.
- **Visualização de Resultados**: Gráficos interativos mostram a proporção entre transações normais e fraudulentas detectadas.
- **Tabela de Resumo**: Exibe um resumo quantitativo das transações monitoradas.

## 📁 Estrutura de Arquivos

- `app-streamlit.py`: Arquivo principal da aplicação Streamlit, responsável por carregar o modelo, processar dados, e exibir a interface.
- `fraud_detection_card.py`: Código Python com o modelo de Machine Learning para detecção de fraudes. Este script treina e salva o modelo `rf_model.pkl` utilizando o dataset de transações de cartão de crédito.
- `creditcard.csv`: Conjunto de dados de transações de cartões de crédito, utilizado para treinar e testar o modelo de detecção de fraudes. Você pode acessar os dados [aqui](https://www.kaggle.com/code/gallo33henrique/ml-creditcard-fraud-lightgbm-93-accuracy?select=creditcard.csv).
- `rf_model.pkl`: Arquivo com o modelo Random Forest treinado, carregado pela aplicação para fazer previsões sobre a natureza (fraudulenta ou normal) das transações.
- `requirements.txt`: Arquivo com as bibliotecas e versões necessárias para execução da aplicação.

## 📥 Como Rodar a Aplicação

Siga os passos abaixo para rodar o projeto em seu ambiente local:

1. **Clone este repositório**:
   ```bash
   git clone git@github.com:eliasvmachado/una-psc-monitoramento-de-ameacas.git
  
3. **Instale as dependências:** Certifique-se de que você está no ambiente virtual correto e execute:
   ```bash
   pip install -r requirements.txt
   
4. **Execute o aplicativo Streamlit:** Para iniciar a aplicação de monitoramento, execute:
   ```bash
   streamlit run app-streamlit.py
> caso não abra automaticamente, acesse http://localhost:8501 para visualizar a aplicação.

## 📈 Exemplo de Uso
Ao iniciar a aplicação, clique em "Iniciar Monitoramento" para começar a análise em tempo real. A aplicação exibirá se cada transação é normal ou fraudulenta, atualizará os contadores de cada tipo e gerará gráficos que mostram a proporção entre transações normais e fraudulentas detectadas.

> by elias victor

