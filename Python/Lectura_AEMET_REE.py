from aemet import Aemet,Estacion
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import json
import requests
import time

# Leemos la clave de la API de AEMET desde un fichero ubicado en el directorio ../API que este notebook

class Ingestion_AEMET:
    
    def __init__(self,path_API='../API/API_KEY_AEMET',path_Data='../Data/'):
        
        self.path_API=path_API
        self.path_Data=path_Data

        with open(self.path_API,'r') as file:
            API_KEY_AE=file.read()

        # Obtenemos el json de estaciones de mediciones de aemet 
        self.info_estaciones=Estacion.get_estaciones(API_KEY_AE)

        # Creamos un objeto Aemet para usar los metodos de la libreria aemet
        self.aemet=Aemet(API_KEY_AE)  
    
    # Definimos funciones que vamos a utilizar para leer los datos de AEMET
    def __estaciones_prov__ (self,lista_estaciones):
        '''Dada una lista de provincias y un json de estaciones de aemet. 
        Obtiene una lista de los ID de las estaciones de esa provincia.'''

        lista_id=[]
        print('Reading list of id of weather stations...')
        for estacion in tqdm(lista_estaciones):
                lista_id.append(estacion['indicativo'])

        return lista_id



    def __lectura_diaria_lista__(self,date_ini,date_end,lista_estaciones):
        '''Dado una lista de id de estaciones de aemet, y fechas de inicio y fin:
        Obtenemos los datos climatologicos entre las dos fechas para todas las estaciones de manera diaria
        Si la fecha de inicio es anterior a 2016, se cambia a 2016-01-01, para evitar errores.
        '''
        valores_diarios=[]
        if date_ini[0:4]<'2016':
            date_ini="2016-01-01T00:00:00UTC"

        if date_ini>date_end:
            print('Valores no válidos, fecha de inicio mayor que la fecha de fin')
            return valores_diarios;

        print("Reading AEMET data from %s to %s ..." % (date_ini,date_end))

        for element in tqdm(lista_estaciones):
            try:
                valores_estacion=self.aemet.get_valores_climatologicos_diarios(date_ini,date_end,element)
                time.sleep(1)
                if type(valores_estacion)!=dict:
                    valores_diarios.extend(valores_estacion)
            except:
                #para evitar errores por nº de lecturas.
                try:
                    valores_estacion=self.aemet.get_valores_climatologicos_diarios(date_ini,date_end,element)
                    time.sleep(1)
                    if type(valores_estacion)!=dict:
                        valores_diarios.extend(valores_estacion)
                except:
                    print('Valor no encontrado')

        return valores_diarios


    def read_weather_dates(self,date_ini,date_end):
        '''Obtenemos los datos de aemet para fechas de mas de un año'''    
        id_estaciones=self.__estaciones_prov__(self.info_estaciones)

        df_weather=pd.DataFrame()

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

            df_weather=df_weather.append(pd.DataFrame(self.__lectura_diaria_lista__(inicio,fin,id_estaciones),dtype=str))

        print('Finish reading AEMET date from %s to %s' % (df_weather['fecha'].min(),df_weather['fecha'].max()) )

        return df_weather
    


    def save_to_csv(self,df):

        path_weather=self.path_Data+'weather.csv'
        df.to_csv(path_weather)
        
        
    def read_from_csv(self):
    
        path_weather=self.path_Data+'weather.csv'
        return pd.read_csv(path_weather,index_col=0,dtype=str)

    
    
    
class Ingestion_REE:
    
    def __init__(self,path_Data='../Data/'):
        
        self.path_Data=path_Data

        # Leemos las regiones de ree obtenidas desde (https://www.ree.es/es/apidatos) desde un fichero ubicado en la misma ruta que este notebook
        self.path_region=self.path_Data+'REGION_REE'
        region_ree=pd.read_csv(self.path_region,header=0,index_col='Region')

        # Me quedo solo con los distintos sistemas electricos existentes
        self.region_system=region_ree[region_ree['geo_limit']!='ccaa']
        
    # Obtenemos los datos de REE a traves de su API. 

    def __lectura_ree_electric_system__(self,d_inicio,d_fin,geo_id):
        '''Dada una fecha'''
        # meter esto en una funcion con su try-exception    
        # Dividir la lectura por años
        reg_filter=self.region_system['geo_id']==geo_id
        
        geo_limit=self.region_system[reg_filter]['geo_limit']

        parametros={'start_date':d_inicio,
                'end_date':d_fin,
                'time_trunc':'day',
                'geo_trunc':'electric_system',
                'geo_limit':geo_limit[0],
                'geo_ids':geo_id}

        URL_GEN='https://apidatos.ree.es/es/datos/generacion/estructura-generacion'

        ree_gen=requests.get(URL_GEN,params=parametros)

        df_ree=pd.DataFrame()
        for i in range(20):
            try:
                df=pd.json_normalize(ree_gen.json()['included'][i]['attributes'],meta=['title','type'],record_path=['values'])
                df['system']=geo_limit[0]
                df_ree=df_ree.append(df)
            except:
                pass #Cuando no hay datos para mas tecnologías
        df_ree.reset_index(inplace=True,drop=True)

        return df_ree
    
    
    def read_ree_dates (self,date_ini,date_end):
    
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
    
        path_ree=self.path_Data+'ree_system.csv'
        df.to_csv(path_ree)
    
    
    def read_from_csv(self):

        path_ree=self.path_Data+'ree_system.csv'
        return pd.read_csv('../Data/ree_system.csv',index_col=0)
        