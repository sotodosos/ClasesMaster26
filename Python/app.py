import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd


features=pd.DataFrame(columns=['system','holidays','fecha','PresMax','PresMin','tmax','tmed','tmin','Prec','velmedia','Racha'])
start_date=dt.date.today()
st.title("Predict Electric Generation from weather")

with st.sidebar:
    st.title('Weather selection')
    with st.form(key='Weather form'):
        weather={}
        err=0
        start_date = st.date_input('Fecha de inicio',value=start_date)
        weather['system'] = st.selectbox('Ree System', ['peninsular', 'canarias','baleares','melilla'])
        weather['fecha']=str(start_date)
        if start_date<dt.date.today():
            st.error('Error: The date have to be greater or equal than today')
            err=1
            
       
        weather['fecha']=str(start_date)
        weather['holidays']= st.selectbox('Nacional holiday:', [1, 0])
        
        weather['PresMax'] = st.slider('Max presure:',min_value=850.0,max_value=1100.0,step=0.1,value=1000.0)
        weather['PresMin'] = st.slider('Min presure:',min_value=850.0,max_value=1100.0,step=0.1,value=1000.0)
        if weather['PresMax']<=weather['PresMin']:
            st.error('Error: PresMax have to be greater than PresMin')
            err=1
            
        weather['tmax'] = st.slider('Max temperature:',min_value=-10.0,max_value=50.0,step=0.01,value=20.0)
        weather['tmed'] = st.slider('Average temperature:',min_value=-15.0,max_value=40.0,step=0.1,value=20.0)
        weather['tmin'] = st.slider('Min temperature:',min_value=-30.0,max_value=40.0,step=0.1,value=20.0)
        if weather['tmax']<=weather['tmed']:
            st.error('Error: Max temperature can´t be lower than average tempature')
            err=1
        elif weather['tmed']<=weather['tmin']:
            st.error('Error: Average temperature can´t be lower than min tempature')
            err=1
        elif weather['tmax']<=weather['tmin']:
            st.error('Error: Max temperature can´t be lower than min tempature')
            err=1
            
        weather['Prec'] = st.slider('Precipitations:',min_value=0.0,max_value=200.0,step=0.01,value=0.0)
        
        weather['velmedia'] = st.slider('Average wind speed:',min_value=0.0,max_value=70.0,step=0.01,value=3.0)
        weather['Racha'] = st.slider('Max wind speed:',min_value=0.0,max_value=140.0,step=0.01,value=10.0)   
        if weather['Racha']<=weather['velmedia']:
            st.error('Error: Average wind speed can´t be lower than max wind speed')
            err=1
            
        submit_button_predict = st.form_submit_button(label='Set weather and predict')

if submit_button_predict and err==0:
    features=features.append(weather,ignore_index=True)
    st.write(features)
    st.map()

