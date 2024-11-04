# 💳 Monitoramento de Fraudes em Cartões de Crédito

## 🎓 Sobre o Projeto
Este é um projeto desenvolvido como parte de uma atividade acadêmica da disciplina de Sistemas Computacionais e Segurança. O objetivo do projeto é criar uma aplicação interativa que utiliza um modelo de Machine Learning para detectar fraudes em transações de cartões de crédito.

O modelo utilizado foi inspirado em um projeto encontrado no Kaggle, que alcançou uma precisão de 93% na detecção de fraudes. O link para o modelo pode ser encontrado [aqui](https://www.kaggle.com/code/gallo33henrique/ml-creditcard-fraud-lightgbm-93-accuracy#-Machine-learning---Credit-Card-Fraud-Detection-).

Através deste projeto, desenvolvemos uma aplicação usando a biblioteca Streamlit para demonstrar os resultados de forma visual e interativa.

## 📊 Funcionalidades
- **Monitoramento em Tempo Real**: A aplicação permite simular a análise de transações em tempo real, identificando transações normais e fraudulentas.
- **Visualização de Resultados**: Gráficos interativos que mostram a proporção de transações normais e fraudulentas.
- **Tabela de Resumo**: Apresentação clara dos resultados com contagens de transações.

## 📁 Estrutura de Arquivos

- `app.py`: O arquivo principal da aplicação Streamlit. Contém a lógica para carregar o modelo, processar os dados e exibir a interface do usuário.
- `creditcard.csv`: Conjunto de dados de transações de cartões de crédito, utilizado para treinar e testar o modelo de detecção de fraudes. Você pode acessar os dados [aqui](https://www.kaggle.com/code/gallo33henrique/ml-creditcard-fraud-lightgbm-93-accuracy?select=creditcard.csv).
- `rf_model.pkl`: O modelo de Machine Learning treinado (Random Forest) que é carregado na aplicação para fazer previsões.
