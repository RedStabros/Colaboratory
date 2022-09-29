import math
import pandas as pd
import matplotlib.pyplot as plt
#Librerias

#Libreria de Machine Learning que generara el modelo
from sklearn import linear_model

#Carga de datos, en este caso ens un csv, pero podria ser cualquier fuente
datos = pd.read_csv(r"C:\Users\Ñañanga\Desktop\TrabajoGSS\Algoritmos\datos_ACPM2.csv")
print(datos)

#Graficamos los puntos de la fuente en el plano
plt.ylabel("Dias ")
plt.xlabel("Mediciones diarias")
plt.scatter(datos["medida"], datos["dia"], color="pink")
#plt.show()

#Generamos el modelo de regresion linear
regresion = linear_model.LinearRegression()

mediciones = datos["medida"].values.reshape((-1, 1))

modelo = regresion.fit(mediciones, datos["dia"])

#el modelo nos devuelve el numero de intercepcion que es el numero de Y cuando X sea 0, osea cuando se termine el ACPM
#la pendiente nos dice la cantidad de dias que necesita el modelo para disminuir 1 litro
print("Interseccion (b)", modelo.intercept_)
print("Pendiente (m)", modelo.coef_)

#Medimos la precision del modelo, mejor puntaje posible 1.0
precision = modelo.score(mediciones, datos["dia"])
print("Precision del Modelo:", precision)

#Entradas de muestra
entrada = [[100],[150],[200],[250],[300]]
modelo.predict(entrada)

#Grafica del modelo generado con la pendiente, los puntos de entrada y la interseccion
plt.scatter(entrada, modelo.predict(entrada), color="red")
plt.plot(entrada, modelo.predict(entrada), color="black")

plt.ylabel("Dias")
plt.xlabel("Mediciones diarias")
plt.scatter(datos["medida"], datos["dia"], color="pink")
#plt.show()

#imprimimos los valores de entrada usados de muestra en el modelo
print(modelo.predict(entrada))

#Este valor de entrada es el nivel actual en litros del tanque de ACPM, a este valor le aplicamos el modelo generado y asi tenemos el estimado de dias faltantes para llegar a 0
entrada2 = [[100]]
modelo.predict(entrada2)
resultado = ((modelo.intercept_)-(modelo.predict(entrada2)))

print("Para", (entrada2), "Litros de ACPM se estiman:", math.ceil(resultado), "dias para terminarse")