import math
import pandas as pd
#import matplotlib.pyplot as plt
#libreria grafico 3D
#from mpl_toolkits.mplot3d import axes3d
#Librerias

#Libreria de Machine Learning que generara el modelo
from sklearn import linear_model, datasets

#Carga de datos, en este caso ens un csv, pero podria ser cualquier fuente
datos = pd.read_csv(r"C:\Users\Ñañanga\Desktop\TrabajoGSS\Algoritmos\datos_ACPM2.csv")
#print(datos)

variablex = datos[["medida", "temp"]]
variabley = datos["dia"]

#Generamos el modelo de regresion linear
regresion = linear_model.LinearRegression()
modelo = regresion.fit(variablex.values, variabley.values)
#el modelo nos devuelve el numero de intercepcion que es el numero de Y cuando X sea 0, osea cuando se termine el ACPM
#la pendiente nos dice la cantidad de dias que necesita el modelo para disminuir 1 litro
print("Interseccion (b)", modelo.intercept_)
print("Pendiente (m)", modelo.coef_)
#Medimos la precision del modelo, mejor puntaje posible 1.0
precision = modelo.score(variablex.values, variabley.values)
print("Precision del Modelo:", precision)
#Este valor de entrada es el nivel actual en litros del tanque de ACPM, a este valor le aplicamos el modelo generado y asi tenemos el estimado de dias faltantes para llegar a 0
entrada2 = [[100, 20]]
modelo.predict(entrada2)
resultado = ((modelo.intercept_)-(modelo.predict(entrada2)))
print("Para", (entrada2), "Litros de ACPM se estiman:", math.ceil(resultado), "dias para terminarse")
