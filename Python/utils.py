

import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def limpieza(df, cols):
    '''
    Inputs:
    :param df:Pandas DataFrame
    :param cols: list of variables
    Output
    :return df: Pandas DataFrame with the variables of cols converted to number
    '''
    for element in cols:
        df[element] = df[element].str.replace(',', '.')
        df[element] = pd.to_numeric(df[element], errors='coerce')
    return df
    

def rellena_nulos_provincia(df, cols_max, cols_min, cols_mean):
    '''
    Create a pandas dataframe with cols_max, cols_min and cols_mean grouped by provincia, fecha, holiday and weekday
    and calculating the max(),min() and mean() for each list .
    Inputs:
    :param df: Pandas Dataframe
    :param cols_max: list of variables to get max values grouped by provincia,fecha,holiday and weekday
    :param cols_min: list of variables to get min values grouped by provincia,fecha,holiday and weekday
    :param cols_mean: list of variables to get mean values grouped by provincia,fecha,holiday and weekday
    Output:
    :return: df_all: Pandas Dataframe with the varibles in the input lists by provincia,fecha,holiday and weekday
    '''

    # Gropued by provincia , fecha, Holiday and weekday, to get de mean,max and min
    df_mean = df.groupby(['provincia', 'fecha', 'Holiday', 'weekday'], as_index=False)[cols_mean].mean()
    df_max = df.groupby(['provincia', 'fecha', 'Holiday', 'weekday'], as_index=False)[cols_max].max()
    df_min = df.groupby(['provincia', 'fecha', 'Holiday', 'weekday'], as_index=False)[cols_min].min()
    # Union of the 3 datasets by the grouped columns
    df_group = pd.merge(df_mean, df_max, how='inner', on=['provincia', 'fecha', 'Holiday', 'weekday'])
    df_all = pd.merge(df_min, df_group, how='inner', on=['provincia', 'fecha', 'Holiday', 'weekday'])

    return df_all


def sns_generacion(df, tech, systems, fecini_zoom, fecfin_zoom):
  '''
    Create a line chart of electric generation by date, tech and system,
    also get a highlight line chart from fecini_zoom to fecfin_zoom
    Inputs:
    :param df: Pandas Dataframe
    :param tech: type of generation technology from df['Technology'] to be represented in the chart
    :param systems: type of generation system from df['system'] that is going to be represented in the chart
    :param fecini_zoom: initial date for highlight chart
    :param fecfin_zoom: end date for highlight chart
    Outputs:
    :return: None (plot a seaborn line chart)
  '''

  # Set the figure
  f = plt.figure(figsize=(15, 32))
  gs = f.add_gridspec(len(systems), 1)
  sns.set_style("ticks")
  sns.set_palette("Set2")

  for i, sys in enumerate(systems):

    #add a subplot for each system in the list
    ax = f.add_subplot(gs[i, 0])
    ax.set_title('Generation evolution for '+sys+' system', fontsize=16)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Generation (Mwh', fontsize=11)

    # Define a filter for df & plot the main lineplot
    filter = (df['Tecnologia'].isin(tech)) & (df['system'] == sys)

    lnplot = sns.lineplot(data=df[filter], x='fecha', y='Generacion_Mwh', ax=ax, legend='full')

    # Define a filter & plot for the highlight lineplot
    filter_covid = filter & (df['fecha'] >= fecini_zoom) & (df['fecha'] <= fecfin_zoom)
    sns.lineplot(data=df[filter_covid], x='fecha', y='Generacion_Mwh', ax=ax)

    # Set a x_ticks with the points of years and dates of covid lockdown
    x_ticks = list(range(0, len(lnplot.get_xticklabels()), 365))
    x_ticks.extend([1519, 1610])
    plt.xticks(x_ticks, ['2016', '2017', '2018', '2019', '2020', '2021', fecini_zoom, fecfin_zoom])
    plt.tick_params('x', labelrotation=-45)
    ax.set_xlim(0)

    # Set y axis limit by electric system
    if sys in ['canarias', 'baleares']:
      ax.set_ylim(5000)
    elif sys == 'peninsular':
      ax.set_ylim(300000)
    else:
      ax.set_ylim(250)

    # quit the chart border
    sns.despine()

  # Set the default palette
  sns.set_palette("Set1")

  return None


def target_preprocesing(df, targets_precentage, systems_names):
    '''
    Calculate the percentages of the columns targets_percentage of Generacion_Mwh in df by electric system and fecha
    Inputs:
    :param df: Pandas dataframe
    :param targets_precentage: list of targets columns to get his percentage over 'Generacion_Mwh
    :param systems_names:  list of electric system to group by
    Output:
    :return df_group: Pandas dataframe with the percentage calculated grouped by electric system and fecha
    '''

    for target in targets_precentage:
        df[target] = (df[target]*df['Generacion_Mwh'])

    systems_names.append('fecha')

    df_group = df.groupby(systems_names, as_index=False).sum()

    for target in targets_precentage:
        df_group[target] = (df_group[target]/df_group['Generacion_Mwh'])
    
    return df_group
    

def date_transform(df):
  '''
  Convert date values (day,month and weekday) to a continuos values. This makes than the difference between
  month 12 and 1 gets the same value that the difference between month 1 and 2.
  For year variable get the log value to get a smaller value, to normalize his weight in our model.
  We don´t use a Scaler() to do this because this penalize the years with less records.

  :param df:  Pandas Dataframe
  :return df: Pandas Dataframe with year,day, month and weekday transformed
  '''

  df['year'] = np.log(df['year'])
  df['day'] = np.cos(((2*np.pi)/31)*df['day'])
  df['month'] = np.cos(((2*np.pi)/12)*df['month'])
  df['weekday'] = np.cos(((2*np.pi)/7)*df['weekday'])

  return df
  

def train_test_val_split(df, features, targets, percentage_test, percentage_val=0.0):
  '''
  Custom train_test_split function, this is created to separate test,train and validation sets from df by date,
  to didn´t use values futures to train our model

  :param df: Pandas Dataframe
  :param features: list of variables of X
  :param targets: list of variables of y
  :param percentage_test: percentage of records of df to get in test set
  :param percentage_val: percentage of records of df to get in test set, by default 0.0
  :return: X_train, X_test, y_train, y_test, X_validation, y_validation
  '''

  # Set the row intervals of each output set
  total = len(df.index)
  row_test = np.round(total-(total*(percentage_test+percentage_val)), 0).astype(np.int)
  row_val = np.round(total-(total*percentage_val), 0).astype(np.int)

  train = df.iloc[:row_test]
  test = df.iloc[row_test:row_val]
  val = df.iloc[row_val:]

  # Separate X and y for each set
  X_train = train[features]
  y_train = train[targets]

  X_test = test[features]
  y_test = test[targets]
  
  X_validation = val[features]
  y_validation = val[targets]

  return X_train, X_test, y_train, y_test, X_validation, y_validation
  
  
def evaluation_function(y_real, y_pred, model):
  '''
  'Calculate the value por each metric (R2,RMSE & MAE) for y_real and y_pred.

  :param y_real: real values of test
  :param y_pred: predicte values to compare with metrics R2,RMSE & MAE
  :param model: Name of the model to try
  :return : Pandas dataframe with the results of the metrics and the name of the model
  '''

  results = {}
  results['Model_name'] = [model]

  # Calculate de MAE metric
  result_mae = mean_absolute_error(y_real, y_pred)
  results['MAE'] = [result_mae]

  # Calculate de RMSE metric
  result_rmse = mean_squared_error(y_real, y_pred, squared=False)
  results['RMSE'] = [result_rmse]

  # Calculate de R2 metric
  results['R2'] = [r2_score(y_real, y_pred)]

  return pd.DataFrame.from_dict(data=results)


def plot_metrics(list_reg):
  '''
  Create a chart for each metric (MAE,RMSE,R2) to compare its scores by models of list_reg

  :param list_reg: list of Pandas Dataframe with name, MAE score, RMSE score & R2 score
  :return: None
  '''

  df = list_reg[0]
  
  for element in list_reg[1:]:
      df = df.append(element)
    
  fig, ax =plt.subplots(3, 1, sharey=True)
  fig.set_size_inches(12, 16)
  fig.suptitle('Comparativa de metricas entre Modelos', fontsize=12)
 
  #MAE
  sns.barplot(y=df['Model_name'], x=df['MAE'], ax=ax[0])
  ax[0].set_title("MAE Compare", fontsize=9)
  
  #RMSE
  sns.barplot(y=df['Model_name'], x=df['RMSE'], ax=ax[1])
  ax[1].set_title("RMSE Compare", fontsize=9)

  #R2
  sns.barplot(y=df['Model_name'], x=df['R2'], ax=ax[2])
  ax[2].set_title("R2 Compare", fontsize=9)

  return None
  
def plot_real_vs_pred(system, target, X_test, y_test, reg_ln, reg_KN, reg_DT, reg_XGB, reg_LGBM):
  '''
    Create a lineplot to compare real vs predicted data for each model and to check if there is overfitting or underfitting,
    also plot the linear model predicted target as naive model.
    
    :param system: electric syste
    :param target: list of targets variables
    :param X_test: Pandas Dataframe
    :param y_test: Real target
    :param reg_ln: linear regressor model fit
    :param reg_KN: k- neighbors regressor model fit
    :param reg_DT: Decision tree model fit
    :param reg_XGB: XGBoostRegressor model fit
    :param reg_LGBM: LightGBMREgressor model fit
    :return: None
  '''
    
  subplt = len(target) * 4
  size = len(target) * 20
  f, ax = plt.subplots(subplt, 1)
  f.set_size_inches(18, size)

  # Create a DataFrame with the results of the predictions of each model
  y_real = pd.DataFrame(data=y_test[y_test[system] == 1])[target]
  y_real['Model'] = 'Real'
  y_real.reset_index(inplace=True)

  y_pred_ln = pd.DataFrame(reg_ln.predict(X_test[X_test[system] == 1]), columns=target)
  y_pred_ln['Model'] = 'Linear Regresor (Base model)'

  y_pred_KN = pd.DataFrame(reg_KN.predict(X_test[X_test[system] == 1]), columns=target)
  y_pred_KN['Model'] = 'KNeighbors'

  y_pred_DT = pd.DataFrame(reg_DT.predict(X_test[X_test[system] == 1]), columns=target)
  y_pred_DT['Model'] = 'Decision Tree'

  y_pred_XGB = pd.DataFrame(reg_XGB.predict(X_test[X_test[system] == 1]), columns=target)
  y_pred_XGB['Model'] = 'XGBoost'

  y_pred_LGBM = pd.DataFrame(reg_LGBM.predict(X_test[X_test[system] == 1]), columns=target)
  y_pred_LGBM['Model'] = 'LightGBM'

  result = pd.concat([y_real, y_pred_ln, y_pred_KN, y_pred_DT, y_pred_XGB, y_pred_LGBM])

  #Make a filter for each model
  models_filter0 = result['Model'].isin(['Real', 'Linear Regresor (Base model)', 'KNeighbors'])
  models_filter1 = result['Model'].isin(['Real', 'Linear Regresor (Base model)', 'Decision Tree'])
  models_filter2 = result['Model'].isin(['Real', 'Linear Regresor (Base model)', 'XGBoost'])
  models_filter3 = result['Model'].isin(['Real', 'Linear Regresor (Base model)', 'LightGBM'])

  sns.set_style("ticks")

  # for each element in target list and each model, make a plot to compare
  for i, element in enumerate(target):

    # To control the subplot to plot each seaborn lineplot, if there are more than 1 variables as target
    if i > 0:
      i = i * 4

    sns.lineplot(data=result[models_filter0], y=element, x=result[models_filter0].index, hue='Model', ax=ax[i])
    sns.lineplot(data=result[models_filter1], y=element, x=result[models_filter1].index, hue='Model', ax=ax[i + 1])
    sns.lineplot(data=result[models_filter2], y=element, x=result[models_filter2].index, hue='Model', ax=ax[i + 2])
    sns.lineplot(data=result[models_filter3], y=element, x=result[models_filter3].index, hue='Model', ax=ax[i + 3])

  sns.despine()

  return None


def chart_altair(df, system_ini='peninsular'):
    '''
    Create an altair chart with the average of last 7 days of electric generation of total enery, renewable energy, solar
    photovoltaic energy and wind powered energy by date, the chart can be filtered by year and electric system.
    also add a vertical line to show the values where put the mouse in the chart.

    :param df: Pandas Dataframe with de average of last 7 days of electric generation by electric system, date, year and technology
    :param system_ini: Initial system to show in the chart, by default 'peninsular
    :return: altair layered chart
    '''

    # labels of X axis to show in the chart, every first day of month from 2016-01 to 2021-12.
    x_labels = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01',
                '2016-07-01', '2016-08-01', '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01', '2016-12-31',
                '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01',
                '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01', '2017-12-31',
                '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01',
                '2018-07-01', '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01', '2018-12-31',
                '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01',
                '2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', '2019-12-31',
                '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01',
                '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2020-12-31',
                '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01',
                '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01', '2021-12-31']

    # list of elements in Color to be plotted
    domain = ['Generación total', 'Renovable', 'Solar fotovoltaica', 'Eólica']
    # colors in hexadecimal, for each element in domain list
    range_ = ['#85C1E9', '#239B56', '#D35400', '#F7DC6F']

    # set a select box to select the system to show in the chart
    select_box_sys = alt.binding_select(options=list(df['system'].unique()))

    selection_sys = alt.selection_single(name='REE',
                                         fields=['system'],
                                         bind=select_box_sys,
                                         init={'system': system_ini})

    # set a radio selector to select the year to show in the chart
    select_radio_year = alt.binding_radio(options=list(df['year'].unique()))

    selection_year = alt.selection_single(name='Choose',
                                          fields=['year'],
                                          bind=select_radio_year,
                                          init={'year': max(df['year'])})

    # create a markpoint with variable fecha as X axis
    # with a selection that works over the variable fecha showing the nearest value where the mouse is over.
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['fecha'], empty='none')

    selectors = alt.Chart(df).mark_point().encode(
        alt.X('fecha'),
        opacity=alt.value(0)
    ).add_selection(
        nearest
    ).transform_filter(
        selection_sys
    ).transform_filter(
        selection_year
    )

    # Create the main chart, with the electric generation by date, and add the selectors of year and systems
    bar = alt.Chart(df[df['Renov_norenov'] == 'Generación total']).mark_area(color='#85C1E9').encode(
        alt.X('fecha', axis=alt.Axis(values=x_labels, labelAngle=0)),
        alt.Y('Generacion_Mwh:Q')

    ).add_selection(
        selection_sys, selection_year
    ).transform_filter(
        selection_sys
    ).transform_filter(
        selection_year
    ).properties(
        width=1400,
        height=450
    )

    # Create the chart of renewable energy by fecha, also add the color with the list domain and his colors in list range_
    # also add the transformers of the main chart
    bar_renov = alt.Chart(df[df['Tecnologia'] == 'Renovable']).mark_area().encode(
        alt.X('fecha'),
        alt.Y('Generacion_Mwh:Q'),
        color=alt.Color('Tecnologia', scale=alt.Scale(domain=domain, range=range_))
    ).transform_filter(
        selection_sys
    ).transform_filter(
        selection_year
    )

    # add a text chart to show the value of the bar_renov chart
    text_renov = bar_renov.mark_text(align='left', dx=3, dy=-20, color='#212F3C').encode(
        text=alt.condition(nearest, 'Generacion_Mwh', alt.value(' '))
    )

    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='fecha',
    ).transform_filter(
        nearest
    )

    # Create the chart of Solar photovoltaic by fecha, also add the transformers of the main chart
    bar_solar = alt.Chart(df[df['Tecnologia'] == 'Solar fotovoltaica']).mark_area(opacity=.8, color='#D35400').encode(
        alt.X('fecha'),
        alt.Y('Generacion_Mwh:Q')
    ).transform_filter(
        selection_sys
    ).transform_filter(
        selection_year
    )

    # add a text chart to show the value of the bar_renov chart
    text_solar = bar_solar.mark_text(align='left', dx=5, dy=-5, color='#212F3C').encode(
        text=alt.condition(nearest, 'Generacion_Mwh', alt.value(' '))
    )

    # Create the chart of wind power by fecha, also add the transformers of the main chart
    bar_eolica = alt.Chart(df[df['Tecnologia'] == 'Eólica']).mark_area(color='#F7DC6F').encode(
        alt.X('fecha'),
        alt.Y('Generacion_Mwh:Q')
    ).transform_filter(
        selection_sys
    ).transform_filter(
        selection_year
    )

    # add a text chart to show the value of the bar_renov chart
    text_eolica = bar_eolica.mark_text(align='left', dx=5, dy=-5, color='#212F3C').encode(
        text=alt.condition(nearest, 'Generacion_Mwh', alt.value(' '))
    )

    # retrun a altair layered chart with all the elements created in the function
    return alt.layer(bar, bar_renov, bar_eolica, bar_solar, selectors, rules, text_renov, text_eolica, text_solar
                    ).configure_axis(labelFontSize=13,titleFontSize=14
                    ).configure_text(fill='#212F3C', fontSize=13
                    ).configure_legend(labelFontSize=14).interactive()
