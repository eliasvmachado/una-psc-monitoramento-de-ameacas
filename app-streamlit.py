import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# Carregar o modelo treinado
model = joblib.load('rf_model.pkl')

# Carregar dados de transação
data = pd.read_csv('creditcard.csv')
data['Class'] = data['Class'].astype(int) 

# Título da página
st.title("Monitoramento em Tempo Real de Transações de Cartão de Crédito")

# desenvolvedores divos
st.markdown(
    """
    <h5 style="text-align: center; color: #8b8b8b; font-weight: normal;">
         Elias Victor, Tayná Mariana, Raissa Maia
    </h5>
    """, 
    unsafe_allow_html=True
)

st.divider()

st.write("""
Aplicativo desenvolvido para a prática da A3 da disciplina de Sistemas
Computacionais e Segurança, com o objetivo de simular o monitoramento
de transações fraudulentas em cartões de crédito.
""")

# Botão para iniciar o monitoramento
start_button = st.empty()  

if start_button.button("Iniciar Monitoramento"):
    
    start_button.empty()

    st.markdown(
        "<p style='text-align: center; color: #777777;'>Monitoramento sendo iniciado...</p>",
        unsafe_allow_html=True
    )
    
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.03)
        progress_bar.progress(percent_complete + 1)
        
    progress_bar.empty()

    # Contadores para fraudes e transações normais
    fraude_count, normal_count = 0, 0

    # Seleção de exemplos da base original
    fraudes_reais = data[data['Class'] == 1].sample(n=5, random_state=1) 
    transacoes_normais = data[data['Class'] == 0].sample(n=15, random_state=1) 
    data_simulada = pd.concat([fraudes_reais, transacoes_normais])
    data_simulada = data_simulada.sample(frac=1).reset_index(drop=True) 

    # Loop para simular a análise em tempo real
    for idx, row in data_simulada.iterrows():
        time.sleep(0.7)

        prediction = model.predict([row.drop('Class').values])[0] 

        # Atualiza contadores e exibe o status
        if prediction == 1:
            fraude_count += 1
            st.error("**Fraude Detectada!**", icon="⚠️")
        else:
            normal_count += 1
            st.success("**Transação Normal**", icon="✅")

        if idx >= 19:
            break

    # Gráficos de análise 
    st.subheader("Análise Geral das Transações")
    fig = go.Figure(data=[go.Pie(labels=["Transações Normais", "Fraudes"],
                                   values=[normal_count, fraude_count],
                                   title="Proporção de Transações Normais vs. Fraudulentas",
                                   hole=.3)])
    st.plotly_chart(fig)

    # Tabela de resumo das transações
    st.subheader("Resumo das Transações")
    summary_data = {
        "Status": ["Total de Transações", "Total de Fraudes", "Total de Normais"],
        "Quantidade": [fraude_count + normal_count, fraude_count, normal_count]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df.style.set_table_attributes('style="font-size: 16px; text-align: center;"'))
    
    st.success("Análise completa. Obrigado por usar nosso aplicativo de monitoramento de fraudes!")