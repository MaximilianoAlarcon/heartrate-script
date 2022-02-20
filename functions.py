import pickle
import numpy as np
import pandas as pd
import sklearn

def leer_archivo_pickle(ruta):
  fileObj = open(ruta, 'rb')
  objeto = pickle.load(fileObj)
  fileObj.close()
  return objeto

prefijo = ''
cluster_predictor = leer_archivo_pickle(prefijo+"cluster_predictor.pkl")
hr_predictor = leer_archivo_pickle(prefijo+"hr_predictor.pkl")

def transformar(dataframe,c):
  prefijo = ''
  paso1 = leer_archivo_pickle(prefijo+'data_transformers/paso1_'+c+'.pkl')
  paso2 = leer_archivo_pickle(prefijo+'data_transformers/paso2_'+c+'.pkl')
  paso3 = leer_archivo_pickle(prefijo+'data_transformers/paso3_'+c+'.pkl')
  paso1.clip = False
  paso3.clip = False
  col = c
  c = dataframe[col].copy()
  c = np.array(c).reshape(-1,1)
  c = paso1.transform(c)
  c = paso2.transform(c)
  c = paso3.transform(c)
  c = np.float32(c)
  dataframe[col] = c


def predecir_senial_hr(df2):
  transformar(df2,"min_capa_verde")
  transformar(df2,"max_capa_verde")
  transformar(df2,"mean_capa_verde")
  transformar(df2,"std_capa_verde")
  df2["CLUSTER"] = cluster_predictor.predict(df2[["min_capa_verde","max_capa_verde","mean_capa_verde","std_capa_verde"]].copy())
  pred = hr_predictor.predict(df2)
  return np.min(pred), np.max(pred), np.mean(pred), np.std(pred), pred