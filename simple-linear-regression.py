""" IA_Parte2.py
    Examen parcial parte 2 donde se calcula el gradiente decendiente
	mínimos cuadrados de error, criterio de paro y al final se grafican

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Thu 23 Feb, 2020
"""
#Importa las librerias estandard
import numpy as np
import matplotlib.pyplot as plt

def criterio_Paro(w_gradiente):
	"""
	Esta funcion es para caluclar el criterio de paro de L2

	INPUTS
	:parametro 1: costo de la w Gradiente

	OUTPUTS
	:return: boleano que indica si para o no

	"""
	threshold= 0.1
	w_gradiente=np.array(w_gradiente)
	norm = np.linalg.norm(w_gradiente)
	return norm <= threshold

def minimos_Cuadrados_Error(y,y_pred,muestra):
	"""
	Esta funcion es para caluclar el minimo cuadrado de error

	INPUTS
	:parametro 1: vector float64
	:parametro 2: vector float64
	:parametro 3: entero con la cantidad de muestras

	OUTPUTS
	:return: float con el minimo cuadrado de error

	"""
	mse= (1/muestra)*sum(total**2 for total in (y-y_pred))
	return mse

def gradiente_Descendiente(x, y, leaning_rate):
	"""
	Esta funcion es para caluclar el gradiente descendiente
	de los datos de entrenamiento

	INPUTS
	:parametro 1: vector float64 valor de la columna x
	:parametro 2: vector float64 valor de la columna y
	:parametro 3: aprendizaje automatico

	OUTPUTS
	:return: diccionario con los valores de w0, w1 y el mse

	"""
	w0 = 0
	w1 = 0

	n = len(x)

	while(True):
	    y_pred = w0 * x + w1
	    w0_gradiente = (1/n)*sum(x*(y_pred-y))
	    w1_gradiente = (1/n)*sum(y_pred-y)
	    if criterio_Paro([w0_gradiente,w1_gradiente]):
	        break
	    w0 = w0 - leaning_rate * w0_gradiente
	    w1 = w1 - leaning_rate * w1_gradiente

	mse = minimos_Cuadrados_Error(y, y_pred, n)
	dataTraining = {'w0': w0, 'w1' : w1, 'mse' : mse}
	return dataTraining

def graficar(x,y,color,color2,dataTraining,dataTesting,type_dataTr,type_dataTes):
	"""
	Esta funcion es para graficar

	INPUTS
	:parametro 1: vector float64 valor de la columna x
	:parametro 2: vector float64 valor de la columna y
	:parametro 3: color distintivo
	:parametro 4: color distintivo
	:parametro 5: diccionario con w0,w1 y mse de los datos de entrenamiento
	:parametro 6: diccionario con w0,w1 y mse de los datos de prueba
	:parametro 7: nombre del tipo de datos
	:parametro 8: nombre del tipo de datos

	OUTPUTS
	:return: dos graficas con los mse dependiendo del tipo de datos que metimos

	"""
	plt.figure(1)
	plt.plot(x, y, '.', color=color, label="Training Data")
	plt.plot(x, dataTraining['w0']*x+dataTraining['w1'],'-g', label="Regression Line")
	plt.legend(loc="upper left")
	plt.title("w0: "+str(dataTraining['w0'])+", w1: "+ str(dataTraining['w1'])+ ", \nMSE("+type_dataTr+"): "+ str(dataTraining['mse']))
	plt.xlabel("Celsius")
	plt.ylabel("Farenheit")

	plt.figure(2)
	plt.plot(x, y, '.', color=color2, label="Testing Data")
	plt.plot(x, dataTesting['w0']*x+dataTesting['w1'],'-g', label="Regression Line")
	plt.legend(loc="upper left")
	plt.title("w0: "+str(dataTesting['w0'])+", w1: "+ str(dataTesting['w1'])+ ", \nMSE("+type_dataTes+"): "+ str(dataTesting['mse']))
	plt.xlabel("Celsius")
	plt.ylabel("Farenheit")
	plt.show()

def imprimir(dataTraining,dataTesting,type_dataTr,type_dataTes,x,x2):
	"""
	Esta funcion es para imprimir la tabla

	INPUTS
	:parametro 5: diccionario con w0,w1 y mse de los datos de entrenamiento
	:parametro 6: diccionario con w0,w1 y mse de los datos de prueba
	:parametro 7: nombre del tipo de datos
	:parametro 8: nombre del tipo de datos
	:parametro 1: vector float64 valor de la columna x de traning
	:parametro 2: vector float64 valor de la columna x de testing
	

	OUTPUTS
	:return: lineas en la terminal donde nos muestra las tablas de traning
	y testing juntos con los MSE

	"""
	print("-"*28)
	print("Data Set Statistics")
	print("-"*28)
	print("Number of samples["+type_dataTr+"]",len(x))
	print("Number of samples["+type_dataTes+"]",len(x2))
	print("-"*28)
	print(type_dataTr)
	print("-"*28)
	print("Tc\t\tTF")
	f = open("tr.txt")
	for linea in f:
		linea=linea.replace(" ", "\t")
		print(linea)
	print("-"*28)
	print("Estimated parameters:")
	print("-"*28)
	print("intercept: ",dataTraining['w1'])
	print("slop: ",dataTraining['w0'],"\n")
	print("-"*28)
	print("Performance Metric ["+type_dataTr+"]")
	print("-"*28)
	print("mean squared error: "+str(dataTraining['mse'])+"\n")
	print("-"*28)

	print(type_dataTes)
	print("-"*28)
	print("Tc\t\tTF")
	f2 = open("tes.txt")
	for linea in f2:
		linea=linea.replace(" ", "\t")
		print(linea)
	print("-"*28)
	print("Performance Metric ["+type_dataTes+"]")
	print("-"*28)
	print("mean squared error: "+str(dataTesting['mse'])+"\n")
	print("-"*28)

def main():
	"""
	Aqui se abren los dos archivos de texto de donde se sacara la informacion
	se determina el learning rate, se saca la gradiante sobre los datos de traning
	y sobre los datos de testing se calcula el mse ya con la w0 y w1 obtenida 
	en la operacion pasada

	Datos de entrada:
	Nada

	Datos de salida:
	Nada
	"""

	type_dataTr= "Training Data"
	type_dataTes= "Testing Data"

	color="blue"

	learning_rate = 0.001

	#Se abre archivo y se gurdan las columnas
	x=[]
	y=[]

	traningDataTxt = open("tr.txt")
	for linea in traningDataTxt:
		data= linea.split(" ")
		x.append(float(data[0]))
		y.append(float(data[1]))


	x = np.array(x,dtype=np.float64)
	y = np.array(y,dtype=np.float64)

	#Metodo para obtener el gradiente descendiente de de los datos de entrenamiento
	dataTraining=gradiente_Descendiente(x,y,learning_rate)

	#Se abre archivo y se gurdan las columnas
	x2=[]
	y2=[]
	testingDataTxt = open("tes.txt")
	for linea in testingDataTxt:
		data= linea.split(" ")
		x2.append(float(data[0]))
		y2.append(float(data[1]))

	x2 = np.array(x2,dtype=np.float64)
	y2 = np.array(y2,dtype=np.float64)

	#Se obtiene el mse de los datos de pruebas usando los w0 y w1 ya obtenidos
	dataTesting={}
	y_pred=dataTraining['w0']*x2+dataTraining['w1']
	muestra=len(x2)
	dataTesting['mse']=minimos_Cuadrados_Error(y2,y_pred,muestra)
	dataTesting['w0']=dataTraining['w0']
	dataTesting['w1']=dataTraining['w1']

	color2="black"

	#Metodo para graficar los resultados
	graficar(x,y,color,color2,dataTraining,dataTesting,type_dataTr,type_dataTes)
	#Metodo para imprimir en terminal los resultados
	imprimir(dataTraining,dataTesting,type_dataTr,type_dataTes,x,x2)

main()

#Doy mi palabra que he realizado este trabajo con Integridad Academica
#codebasics. (2018, July 22). Machine Learning Tutorial Python - 4: Gradient Descent and Cost Function [Video file]. Retrieved from https://www.youtube.com/watch?v=vsWrXfO3wWw