import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
import utils
import joblib

features=pd.DataFrame(columns=['day','year','month','system','Holiday','weekday','tmin','tmed','presMin','prec','sol','velmedia','tmax','presMax','racha'])
start_date=dt.date.today()
st.title("Predict Electric Generation from weather")
path_model='../models/'
with st.sidebar:
    st.title('Weather selection')
    with st.form(key='Weather form'):
        weather={}
        err=0
        start_date = st.date_input('Fecha de inicio',value=start_date)
        weather['system'] = st.selectbox('Ree System', ['peninsular', 'canarias','baleares','melilla'])
        date=str(start_date)
        list_date=date.split('-')
        if start_date<dt.date.today():
            st.error('Error: The date have to be greater or equal than today')
            err=1
            
        
        weather['year']=float(list_date[0])
        weather['month']=float(list_date[1])
        weather['day']=float(list_date[2])
        weather['weekday']=float(start_date.weekday())
        
        holidays= st.radio('Nacional holiday:', ['No','Yes'])
        if holidays=='No':
            weather['Holiday']=0
        else: 
            weather['Holiday']=1
            
        weather['presMax'] = st.slider('Max presure:',min_value=850.0,max_value=1100.0,step=0.1,value=1000.0)
        weather['presMin'] = st.slider('Min presure:',min_value=850.0,max_value=1100.0,step=0.1,value=1000.0)
        if weather['presMax']<=weather['presMin']:
            st.error('Error: Max presure have to be greater than Min presure')
            err=1
        
        weather['sol'] = st.slider('Sun hours:',min_value=-0.0,max_value=15.0,step=1.0,value=8.0)
        weather['tmax'] = st.slider('Max temperature:',min_value=-30.0,max_value=50.0,step=0.1,value=20.0)
        weather['tmed'] = st.slider('Average temperature:',min_value=-30.0,max_value=50.0,step=0.1,value=15.0)
        weather['tmin'] = st.slider('Min temperature:',min_value=-30.0,max_value=50.0,step=0.1,value=10.0)
        if weather['tmax']<=weather['tmed']:
            st.error('Error: Max temperature can´t be lower than average tempature')
            err=1
        elif weather['tmed']<=weather['tmin']:
            st.error('Error: Average temperature can´t be lower than min tempature')
            err=1
        elif weather['tmax']<=weather['tmin']:
            st.error('Error: Max temperature can´t be lower than min tempature')
            err=1
            
        weather['prec'] = st.slider('Precipitations:',min_value=0.0,max_value=200.0,step=0.01,value=0.0)
        
        weather['velmedia'] = st.slider('Average wind speed:',min_value=0.0,max_value=70.0,step=0.01,value=3.0)
        weather['racha'] = st.slider('Max wind speed:',min_value=0.0,max_value=140.0,step=0.01,value=10.0)   
        if weather['racha']<=weather['velmedia']:
            st.error('Error: Average wind speed can´t be lower than max wind speed')
            err=1
            
        submit_button_predict = st.form_submit_button(label='Set weather and predict')

if submit_button_predict and err==0:
    features=features.append(weather,ignore_index=True)
    st.dataframe(features)
    st.dataframe(features.info())
    
    reg_renov = joblib.load('best_model_renovable.sav')
    pred_renov=reg_renov.predict(features)
    st.write(pred_renov)
    
    reg_generation = joblib.load('best_model_generation.sav')
    pred_generation=reg_generation.predict(features)
    st.write(pred_generation)
    
    reg_tech = joblib.load('best_model_tech.sav')
    pred_tech=reg_tech.predict(features)
    st.write(pred_tech)

