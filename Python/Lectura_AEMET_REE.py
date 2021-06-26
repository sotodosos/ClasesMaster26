from aemet import Aemet, Estacion
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import requests
import time

class Ingestion_AEMET:
    '''
    Class for read, clean and save the data from AEMET OpenData. To use this class it´s required to have the API Key from aemet.
    '''
    def __init__(self, path_API='../API/API_KEY_AEMET', path_Data='../Data/'):
        '''
        Initialize the object class. By default it takes the API key from ../API/API_KEY_AEMET
        and the read and save the data from ../Data/

        :param path_API: path of the API Key
        :param path_Data:  path of the data to read and to save .
        '''

        self.path_API = path_API
        self.path_Data = path_Data

        with open(self.path_API, 'r') as file:
            API_KEY_AE = file.read()

        # Obtain the json of stations
        self.info_estaciones = Estacion.get_estaciones(API_KEY_AE)

        # Create an Aemet object to read data.
        self.aemet = Aemet(API_KEY_AE)
    
    # Custom functions
    def __estaciones_prov__(self, lista_estaciones):
        '''
        With a list of aemet stations , get a list of station IDs '

        :param lista_estaciones: list of aemet weather stations
        :return: list of id of the aemet weather stations
        '''

        lista_id = []
        print('Reading list of id of weather stations...')
        for estacion in tqdm(lista_estaciones):
            lista_id.append(estacion['indicativo'])

        return lista_id

    def __lectura_diaria_lista__(self, date_ini, date_end, lista_estaciones):
        '''
        With a list of the Aemet station ids and the start and end dates:
           Obtains the weather data between the two dates for all the stations on a daily basis
           If the start date is before 2016, it is changed to 2016-01-01, to avoid errors.

        :param date_ini: Initial date to get the data
        :param date_end: Finish date to get the data
        :param lista_estaciones: list of the Aemet station ids
        :return: weather data between the two dates for all the stations ids
        '''

        valores_diarios = []
        # To avoid errors if the start date is before 2016 it is changed to 2016-01-01
        if date_ini[0:4] < '2016':
            date_ini = "2016-01-01T00:00:00UTC"

        if date_ini > date_end:
            print('Valores no válidos, fecha de inicio mayor que la fecha de fin')
            return valores_diarios

        print("Reading AEMET data from %s to %s ..." % (date_ini, date_end))

        # For each id execute the request function from aemet library "get_valores_climatologicos_diarios"
        # to get the weather data between date_ini and date_end
        for element in tqdm(lista_estaciones):
            try:
                valores_estacion = self.aemet.get_valores_climatologicos_diarios(date_ini, date_end, element)
                # To avoid errors of number of requests, it takes 1 sec between requests
                time.sleep(1)
                if type(valores_estacion) != dict:
                    valores_diarios.extend(valores_estacion)
            except:
                #To avoid errors of number of requests, why retry the request after sleep 56 seconds.
                try:
                    time.sleep(56)
                    valores_estacion=self.aemet.get_valores_climatologicos_diarios(date_ini,date_end,element)
                    if type(valores_estacion) != dict:
                        valores_diarios.extend(valores_estacion)
                except:
                    print('Valor no encontrado')

        return valores_diarios


    def read_weather_dates(self, date_ini, date_end):
        '''
        Obtains aemet data for dates of more than one year, this is used for have more control of the date readed
        and to convert the list returne by __lectura_diaria_lista__ to Pandas Dataframe.

        :param date_ini: Initial date to read data
        :param date_end: Finish date to read data
        :return: pondas Dataframe
        '''

        id_estaciones = self.__estaciones_prov__(self.info_estaciones)

        df_weather = pd.DataFrame()

        year_ini = int(date_ini[0:4])
        year_fin = int(date_end[0:4])

        for i in np.arange(year_ini, year_fin+1):
            if i == year_ini:
                inicio = date_ini
                fin = str(i)+"-12-31T00:00:00UTC"

            elif i == year_fin:
                inicio = str(i)+"-01-01T00:00:00UTC"
                fin = date_end

            else:
                inicio = str(i)+"-01-01T00:00:00UTC"
                fin = str(i)+"-12-31T00:00:00UTC"

            df_weather = df_weather.append(pd.DataFrame(self.__lectura_diaria_lista__(inicio, fin, id_estaciones), dtype=str))

        print('Finish reading AEMET date from %s to %s' % (df_weather['fecha'].min(), df_weather['fecha'].max()) )

        return df_weather

    def save_to_csv(self, df):
        '''
        Saves DataFrame to 1_weather.csv
        :param df: Datarame to save
        :return:
        '''
        path_weather=self.path_Data+'1_weather.csv'
        df.to_csv(path_weather)

    def read_from_csv(self):
        '''
        Read 1_weather.csv
        :return: Pandas DataFrame
        '''
    
        path_weather=self.path_Data+'1_weather.csv'
        return pd.read_csv(path_weather, index_col=0, dtype=str)
    
    
class Ingestion_REE:
    '''
        Class for read, clean and save the data from REE,
    '''
    def __init__(self, path_Data='../Data/'):
        '''
        Initialize the object class. By default it takes the data from ../Data/

        :param path_Data:  path of the data to read and to save .
        '''

        self.path_Data=path_Data

        # Read the ree regions obtained from (https://www.ree.es/es/apidatos) from a file located in path_Data
        self.path_region=self.path_Data+'REGION_REE'
        region_ree=pd.read_csv(self.path_region,header=0,index_col='Region')

        # Get only with the different existing electrical systems
        self.region_system=region_ree[region_ree['geo_limit']!='ccaa']

    def __lectura_ree_electric_system__(self,d_inicio,d_fin,geo_id):
        '''
        Reading data from ree daily data and electric system from REE API (https://www.ree.es/es/apidatos),
        the request returns a json, which is organized by technology

        :param d_inicio: Initial date to read
        :param d_fin: last date to read
        :param geo_id: region to read data from region_system
        :return: Pandas dataframe with the daily data for de region geo_id and every technology
        '''
        reg_filter=self.region_system['geo_id']==geo_id

        # Get the geo_limit for the request
        geo_limit=self.region_system[reg_filter]['geo_limit']

        # Set a dictionary to use in the request
        parametros={'start_date':d_inicio,
                'end_date':d_fin,
                'time_trunc':'day',
                'geo_trunc':'electric_system',
                'geo_limit':geo_limit[0],
                'geo_ids':geo_id}

        URL_GEN='https://apidatos.ree.es/es/datos/generacion/estructura-generacion'

        # Execute the request to REE
        ree_gen=requests.get(URL_GEN,params=parametros)

        df_ree=pd.DataFrame()
        # Read the json from the request and transform to Pandas Dataframe
        # It takes a loop for each technology, and there are 17 tech. It will do 20 loops because maybe REE can include new technologies.

        for i in range(20):
            try:
                df=pd.json_normalize(ree_gen.json()['included'][i]['attributes'],meta=['title','type'],record_path=['values'])
                df['system']=geo_limit[0]
                df_ree=df_ree.append(df)
            except:
                pass # If there are not more data, do nothing.
        df_ree.reset_index(inplace=True,drop=True)

        return df_ree
    
    
    def read_ree_dates (self,date_ini,date_end):
        '''
        Obtains REE data for dates of more than one year, execute __lectura_ree_electric_system__ one time per electric system.

        :param date_ini: Initial date to read REE data
        :param date_end: Finish date to read REE data
        :return: Pandas Dataframe
        '''

        df_ree_system=pd.DataFrame()
        
        year_ini=int(date_ini[0:4])
        year_fin=int(date_end[0:4])
        
        for i in np.arange(year_ini,year_fin+1):
            if i==year_ini:
                inicio=date_ini
                fin=str(i)+"-12-31T00:00:00UTC"

            elif i==year_fin:
                inicio=str(i)+"-01-01T00:00:00UTC"
                fin=date_end

            else:
                inicio=str(i)+"-01-01T00:00:00UTC"
                fin=str(i)+"-12-31T00:00:00UTC"

        
            for electric_system in tqdm(self.region_system['geo_id']):
                df_ree_system=df_ree_system.append(self.__lectura_ree_electric_system__(inicio,
                                                                                        fin,
                                                                                        electric_system))
        
        return df_ree_system 
    
    
    def save_to_csv(self,df):
        '''
        Saves DataFrame to ree_system.csv
        :param df: Datarame to save
        :return:
        '''
        path_ree=self.path_Data+'1_ree_system.csv'
        df.to_csv(path_ree)
    
    
    def read_from_csv(self):
        '''
        Read ree_system.csv
        :return: Pandas DataFrame
        '''
        path_ree=self.path_Data+'1_ree_system.csv'
        return pd.read_csv(path_ree,index_col=0)
        