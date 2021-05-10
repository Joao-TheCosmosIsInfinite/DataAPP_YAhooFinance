# Bibliotecas
import numpy as numpy
import yfinance as yf 
import streamlit as st
import matplotlib.pyplot as plt  
from fbprophet import Prophet 
from fbprophet.plot import plot_plotly 
from plotly import graph_objs as go 
from datetime import date 
import warnings
warnings.filterwarnings('ignore')

# Define a data de inicio e fim para coleta de dados
DT_BEGIN = '2015-01-01'
DT_TODAY = date.today().strftime('%Y-%m-%d')

# Definir o titulo do dashboard
st.title('Dashboard Financeiro para Previsão de Ativos Financeiros')

# Definir as empresas para coleta de dados
company_name = ('PBR', 'GOOG', 'PFE', 'UBER', 'AZN', 'XP', 'MGLU3.SA')


# PBR         - Petrobras 
# GOOG        - Google 
# PFE         - Pfizer 
# UBER        - Uber 
# AZN         - Astrazeneca 
# XP          - XP Investimentos 
# MGLU3.SA    - Magazine Luiza 



# Define de qual empresa usaremos os dados por vez
selected_company = st.selectbox('Selecione a Empresa para as Previsoes de Ativos Financeiros', company_name)


@st.cache
def load_data (ticker):
    '''
    
    '''
    data = yf.download(ticker, DT_BEGIN, DT_TODAY)
    data.reset_index(inplace = True)
    return data

# Mensagem
message = st.text('Carregando Dados')

# Carregando os dados
data = load_data(selected_company)

# Mensagem de encerramento da carga de dados
message.text('Carregando os dados ... Concluído')

# Sub-titulo
st.subheader('Visualizacao de Dados Brutos')
st.write(data.tail(10))

# Funcao para a plotagem
def plot_data():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = data['Date'], 
                             y = data['Open'], 
                             name = 'stock_open'))

    fig.add_trace(go.Scatter(x = data['Date'], 
                             y = data['Close'], 
                             name = 'stock_close'))

    fig.layout.update(title_text = 'Preco de Abertura e Fechamento das Acoes', xaxis_rangeslider_visible = True)

    st.plotly_chart(fig)

# Executando a Funcao
plot_data()

st.subheader('Previsoes com Machine Learning')

# Criando os dataframes com os dados que sera processados
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {'Date': 'ds',
                                     'Close': 'y'})

# Cria o modelo
model = Prophet()

# Treina o modelo
model.fit(df_train)

# Define o horizonte de previsao
num_year = st.slider('Horizonte de Previsao (em anos):', 1, 4)

# Calcula o período em dias
period = num_year * 365

# PRepara as datas futuras para as previsoes
future = model.make_future_dataframe(periods = period)

# Faz as previsoes
forecast = model.predict(future)

# Subtitulo
st.subheader('Dados Previstos')

# Dados Previstos
st.write(forecast.tail(10))

# Titulo
st.subheader('Previsao de Preco dos Ativos Financeiros para o Periodo Selecionado')

# Plot
grafico2 = plot_plotly(model, forecast)
st.plotly_chart(grafico2)