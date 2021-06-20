import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from utils import chart_altair

cols = ['day', 'year', 'month', 'system', 'Holiday', 'weekday', 'tmin','tmed', 'presMin', 'prec', 'sol', 'velmedia', 'tmax', 'presMax', 'racha']
cols_renov= ['day', 'year', 'month', 'system', 'Holiday', 'weekday', 'tmin', 'presMin', 'prec', 'sol', 'velmedia', 'tmax', 'presMax', 'racha']


features = pd.DataFrame(columns=cols)
path_model = '../models/'
path_data = '../Data/'

st.set_page_config(page_title='Electric Generation',layout='wide')
st.title("Predict Electric Generation from weather")




with st.sidebar:
    st.title('Weather selection')
    with st.form(key='Weather form'):
        weather = {}
        err = 0
        start_date = dt.date.today()
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
        weather['presMin'] = st.slider('Min pressure:', min_value=850.0, max_value=1100.0, step=0.1, value=900.0)
        if weather['presMax'] <= weather['presMin']:
            st.error('Error: Max pressure have to be greater than Min pressure')
            err = 1
        
        weather['sol'] = st.slider('Sun hours:', min_value=-0.0, max_value=15.0, step=1.0, value=8.0)
        weather['tmax'] = st.slider('Max temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=20.0)
        weather['tmed'] = st.slider('Average temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=15.0)
        weather['tmin'] = st.slider('Min temperature:', min_value=-30.0, max_value=50.0, step=0.1, value=10.0)
        if weather['tmax'] <= weather['tmed']:
            st.error('Error: Max temperature can´t be lower than Average temperature')
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
    perc_renov = reg_renov.predict(features[cols_renov])
        
    reg_generation = joblib.load(path_model+'best_model_Generation.sav')
    value_generation = reg_generation.predict(features)

    reg_tech = joblib.load(path_model+'best_model_tech.sav')
    value_tech = reg_tech.predict(features[cols_renov])
    value_renov = perc_renov*value_generation
    perc_norenov = 1-perc_renov
    prec_norenov_str = str(round(perc_norenov[0] * 100, 2))
    prec_renov_str = str(round(perc_renov[0]*100,2))
    value_norenov = value_generation-value_renov
    perc_tech1 = value_tech[:,0]
    value_tech1 = perc_tech1*value_generation
    perc_tech1_str = str(round(perc_tech1[0] * 100, 2))
    perc_tech2 = value_tech[:,1]
    value_tech2 = perc_tech2 * value_generation
    perc_tech2_str = str(round(perc_tech2[0]*100,2))
    perc_other = perc_renov - perc_tech1 - perc_tech2

    if perc_other>0:
        value_other = perc_other * value_generation
        perc_tech_ot_str = str(round(perc_other[0]*100,2))
    else:
        value_other = [0]
        perc_tech_ot_str = '0.00'
        perc_other = 0

    st.write('---------------------------------------------')

    st.header('Predicted values')
    st.write('\n')

    c1, c2 = st.beta_columns((4, 5))
    with c2:
        st.subheader('Explanation')
        st.write('\n')
        st.write('For '+str(start_date)+' in '+weather['system']+' system our model predict a total generation of '+str(round(value_generation[0],3))+' Mwh.')
        st.write('That is divided into '+str(round(value_norenov[0],3))+' Mwh ('+prec_norenov_str+'%) no renewable energy ')
        st.write('and '+str(round(value_renov[0],3))+' Mwh ('+prec_renov_str+'%) renewable energy.')
        st.write('\n')
        st.write('Renewal energy is distributed in:')
        st.write('- Wind power:'+str(round(value_tech2[0],3))+ ' Mwh ('+perc_tech2_str+'%)')
        st.write('- Solar photovoltaic: '+str(round(value_tech1[0],3))+' Mwh ('+perc_tech1_str+'%)')
        st.write('- Others: '+str(round(value_other[0],3))+' Mwh ('+perc_tech_ot_str+'%)')

    with c1:
        st.write('\n')
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots()

        size = .3
        radius=.9
        angle=45
        vals = np.array([[perc_norenov,0,0], [perc_tech1,perc_tech2,perc_other]])

        cmap = plt.get_cmap("tab20c")
        outer_colors = cmap([5, 9])
        inner_colors = cmap([6, 7, 8, 9, 10, 11])

        ax.pie(vals.sum(axis=1), radius=radius, colors=outer_colors,
               labels=[prec_norenov_str+'% No Renewable ',prec_renov_str+'% Renewable'],
               labeldistance=.8,startangle=angle,
               wedgeprops=dict(width=size, edgecolor='w',linewidth=3))

        ax.pie(vals.flatten(), radius=radius - size, colors=inner_colors,
               labels=['','','',perc_tech1_str+'% Solar',perc_tech2_str+'% Wind',perc_tech_ot_str+'% Other'],
               labeldistance=.5,startangle=angle,
               wedgeprops=dict(width=size, edgecolor='w',linewidth=1.5))

        ax.set(aspect="equal", title='Distribution of predicted generation')

        st.pyplot(fig)

    st.write('---------------------------------------------')
    st.header('History distribution of electric generation')
    st.write("\n")
    df_rolling = pd.read_pickle(path_data + 'Rolling_chart.pkl')
    st.altair_chart(chart_altair(df_rolling))