import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

cols = ['day', 'year', 'month', 'system', 'Holiday', 'weekday', 'tmin', 'tmed', 'presMin', 'prec', 'sol', 'velmedia', 'tmax', 'presMax', 'racha']
features = pd.DataFrame(columns=cols)
start_date = dt.date.today()
st.title("Predict Electric Generation from weather")
path_model = '../models/'

with st.sidebar:
    st.title('Weather selection')
    with st.form(key='Weather form'):
        weather = {}
        err = 0
        start_date = st.date_input('Fecha de inicio', value=start_date)
        weather['system'] = st.selectbox('Ree System', ['peninsular', 'canarias', 'baleares', 'melilla'])
        date = str(start_date)
        list_date = date.split('-')
        if start_date < dt.date.today():
            st.error('Error: The date have to be greater or equal than today')
            err = 1

        weather['year'] = float(list_date[0])
        weather['month'] = float(list_date[1])
        weather['day'] = float(list_date[2])
        weather['weekday'] = float(start_date.weekday())
        
        holidays = st.radio('National holiday:', ['No', 'Yes'])
        if holidays == 'No':
            weather['Holiday'] = 0
        else: 
            weather['Holiday'] = 1
            
        weather['presMax'] = st.slider('Max pressure:', min_value=850.0, max_value=1100.0, step=0.1, value=1000.0)
        weather['presMin'] = st.slider('Min pressure:', min_value=850.0, max_value=1100.0, step=0.1, value=1000.0)
        if weather['presMax'] <= weather['presMin']:
            st.error('Error: Max pressure have to be greater than Min pressure')
            err = 1
        
        weather['sol'] = st.slider('Sun hours:', min_value=-0.0, max_value=15.0, step=1.0, value=8.0)
        weather['tmax'] = st.slider('Max temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=20.0)
        weather['tmed'] = st.slider('Average temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=15.0)
        weather['tmin'] = st.slider('Min temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=10.0)
        if weather['tmax'] <= weather['tmed']:
            st.error('Error: Max temperature can´t be lower than average temperature')
            err = 1
        elif weather['tmed'] <= weather['tmin']:
            st.error('Error: Average temperature can´t be lower than min temperature')
            err = 1
        elif weather['tmax'] <= weather['tmin']:
            st.error('Error: Max temperature can´t be lower than min temperature')
            err = 1
            
        weather['prec'] = st.slider('Precipitations:', min_value=0.0, max_value=200.0, step=0.01, value=0.0)
        
        weather['velmedia'] = st.slider('Average wind speed:', min_value=0.0, max_value=70.0, step=0.01, value=3.0)
        weather['racha'] = st.slider('Max wind speed:', min_value=0.0, max_value=140.0, step=0.01, value=10.0)
        if weather['racha'] <= weather['velmedia']:
            st.error('Error: Average wind speed can´t be lower than max wind speed')
            err = 1
            
        submit_button_predict = st.form_submit_button(label='Set weather and predict')

if submit_button_predict and err == 0:
    features = features.append(weather, ignore_index=True)

    reg_renov = joblib.load(path_model+'best_model_renovable.sav')
    perc_renov = reg_renov.predict(features)
        
    reg_generation = joblib.load(path_model+'best_model_Generation.sav')
    value_generation = reg_generation.predict(features)

    reg_tech = joblib.load(path_model+'best_model_tech.sav')
    value_tech = reg_tech.predict(features)

    value_renov = perc_renov*value_generation
    perc_norenov = 1-perc_renov
    value_norenov = value_generation-value_renov
    perc_tech = value_tech
    perc_other = perc_renov - perc_tech
    st.write('---------------------------------------------')
    st.write('History')
    st.write('---------------------------------------------')
    st.write('Generación Total:')
    st.write(value_generation)
    st.write('Porcentaje de energía renovable:')
    st.write(perc_renov*100)
    st.write('Valores de energía renovable y no renovable:')
    st.write(value_renov)
    st.write(value_norenov)
    st.write('Porcentaje de energia Solar Fotovoltaica :')
    st.write(perc_tech*100)
    st.write('Valor de energia Solar Fotovoltaica :')
    st.write(value_tech)

    st.write('---------------------------------------------')

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()

    size = .3
    radius=.9
    angle=90
    vals = np.array([[perc_norenov,0], [perc_tech,perc_other]])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap([5, 9])
    inner_colors = cmap([6, 7, 10,11,12])

    ax.pie(vals.sum(axis=1), radius=radius, colors=outer_colors,
           labels=[str(round(perc_norenov[0]*100,2))+'% No Renewable ',str(round(perc_renov[0]*100,2))+'% Renewable'],
           labeldistance=.8,startangle=angle,
           wedgeprops=dict(width=size, edgecolor='w',linewidth=3))

    ax.pie(vals.flatten(), radius=radius - size, colors=inner_colors,
           labels=['','',str(round(perc_tech[0]*100,2))+'% Solar',str(round(perc_other[0]*100,2))+'% Other'],
           labeldistance=.45,startangle=angle,
           wedgeprops=dict(width=size, edgecolor='w',linewidth=1.5))

    ax.set(aspect="equal", title='Distribution of predicted generation')


    st.pyplot(fig)