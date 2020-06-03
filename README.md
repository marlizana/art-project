<img src="https://www.sketch.ca/sketchPub/uploads/2019/03/radical-art-of-young-people-2000x940.jpg" alt="ART" width="1000" height="200" align="center"/>

# Projecto final: Arte


Mar Lizana Atienza

*Data Part Time Barcelona Dic 2019*


## Content

**Índice**   
1. [Project Description](#id1)
2. [Dataset](#id2)
3. [Workflow](#id3)
4. [Results](#id4)
5. [Bibliografía](#id5)

<hr style="color: #7acaff;" width="50%" />

<a name="project"></a>

## Project Description<a name="id1"></a>

Mediante la aplicación de técnicas de deep-learning entrenaré un algoritmo capaz de identificar los autores de obras de arte. Los motivos principales de la selección de este tema era poder trabajar con imágenes y redes neuronales.

<hr style="color: #7acaff;" width="50%" />
<a name="dataset"></a>

## Dataset<a name="id2"></a>

Hemos seleccionado un dataset de la plataforma Kaggle sobre <a href="https://www.kaggle.com/ikarus777/best-artworks-of-all-time">los 50 artistas más influyentes</a> de la historia del arte. Se trata de una colección de imágenes de las obras de arte y un conjunto de datos con información básica recuperada de wikipedia en formato CSV. 

* **artists.csv**:

<CODE> id </CODE>: Identificador del artista.

<CODE> name </CODE> : Nombre del artista.

<CODE>years</CODE> : Año de nacimiento y defunción del artista.

<CODE>genre</CODE> : Movimiento/s en el que se le incluye.

<CODE>nationality</CODE> : Nacionalidad.

<CODE>bio</CODE> : Biografía.

<CODE>wikipedia</CODE> : URL a la entrada de Wikipedia.

<CODE>paintings</CODE> : número total de pinturas adjuntadas.
    

* **images.zip**:
    Colección de 8446 imágenes de cuadros de 50 artistas.
    
    
* **resized.zip**:
    Subset con imágenes de menor calidad para que los modelos vayan mejor.

<hr style="color: #7acaff;" width="50%" />
<a name="workflow"></a>

## Workflow<a name="id3"></a>

El recorrido del proyecto es el siguiente:


1. [Análisis del proyecto](#id6)
2. [EDA](#id7)
3. [Data Wrangling](#id8)
4. [Choosing a Model](#id9)
5. [Fine Tuning](#id10)
6. [Sistema de recomendación(EXTRA)](#id111)

### Análisis del proyecto<a name="id6"></a>

El trabajar con datos que requieren una capacidad computacional superior a mi ordenador requería de los servicios que ofrece Google en la nube. Cree dos máquinas virtuales, la primera con 30 GB de memoria y una GPU NVIDIA K80 y una segunda con 60 GB de memoria.

### EDA<a name="id7"></a>

Para el **análisis exploratorios de los datos** empecé trabajando con el csv. generé un grafo que me permitiera ver las relaciones entre los pintores através de los movimientos artísticos.

<img src="img/grafo.png" alt="Grafo"/>

Generé una primera visualización de las imágenes. Aplicando la función <code>open_images_names</code> obtendremos las imágenes a analizar junto a una lista con los nombres que se extraen del archivo, y solo habrá que pasarle la dirección del directorio donde están nuestras imagenes junto con <k>/**</k> para que extraiga todos los archivos que hay dentro.

<img src="img/random_pictures.png" alt="Random"/>

Con la función <code>muestra</code> podemos ver una imagen al azar de nuestra colección junto al nombre del artista y el tamaño. Cada vez que se ejecute saldrá una diferente.

<img src="img/ejemplo2.jpeg" alt="ejemplo_muestra"/>

Podemos observar por ejemplo que disponemos de más de 800 cuadros de Van Gogh, frente a unos 50 de Cezanne. Hay diferentes estrategias que podemos seguir para corregir este problema:

<img src="img/countplot50.jpeg" alt="countplot50"/>

* Empezaremos probando con una selección de artistas que tengan el número de obras más balanceadas. Primero con una selección de los 5 pintores con más registros para luego ir ampliando.

<img src="img/countplot5.jpeg" alt="countplot5"/>
<img src="img/countplot10.jpeg" alt="countplot10"/>


### Data Wrangling<a name="id8"></a>

#### Ajustar el tamaño de las imágenes

Las imágenes que forman nuestra colección tienen diferentes tamaños; deberemos establecer un tamaño común para poder entrenar nuestro algoritmo. En este caso, limitaremos el tamaño a una dimensión de 100px para que los modelos puedan ejecutarse más rápido. Usaremos la función creada <code>resize_data</code> para ello.

<img src="img/standar100.jpeg" alt="standar100"/>

Claramente la imagen a perdido bastante calidad, veamos como reaccionan los modelos.

#### DataAugmentation

Ahora estableceremos los parámetros para la creación de nuevas imagenes a partir de modificaciones de la original. Estas variaciones permiten aprovechar cada parte de la imagen para encontrar los patrones, lo cual es muy útil cuando tenemos pocas imágenes y para que el modelo generalice mejor. Podemos realizar las siguientes modificaciones:
* **rotation_range**: Indica el numero maximo de grados que la imagen se puede inclinar.
* **width shift range, height shift range**: cambia de orientación los pixeles de algunas partes de la imagen.
* **shear_range**: Modifica algunas partes de la imagen modificando la orientación.
* **zoom_range**: Aplica un acercamiento a la imagen.
* **horizontal_flip**: Cambia la orientación de la imagen.
* **fill_mode**: Cuando a la imagen se le aplica una rotación cambia su aspecto, para mantener el mismo aspecto se tienen que rellenar los pixeles faltantes, con la opción nearest los pixeles cercanos se repiten para rellenar las areas faltantes.

<img src="img/dataugmentation.jpeg" alt="full"/>
<img src="img/dataugmentation1.jpeg" alt="ej1"/>
<img src="img/dataugmentation2.jpeg" alt="ej2"/>

### Elegir el modelo<a name="id9"></a>

#### Algunas consideraciones previas

**¿Qué es una red neuronal?**

La idea principal que hay detrás de las **redes neuronales** es la de imitar el funcionamiento de las de los organismos vivos: un conjunto de neuronas conectadas entre sí que trabajan en conjunto. Con la experiencia, las neuronas van creando y reforzando ciertas conexiones para "aprender". La información de entrada atraviesa la neurona, donde se llevan a cabo determinadas operaciones, produciendo unos valores de salida.

Las redes neuronales son un modelo para encontrar esa combinación de parámetros y aplicarla al mismo tiempo. Una red ya entrenada se puede usar luego para hacer predicciones o clasificaciones con otros datos.

<img src="img/red ejemplo.png" alt="red"/>

Antes de comenzar a probar modelos determinaremos nuestros objetivos en los resultados. Para ello definimos las métricas que nos indicarán si nuestro modelo está haciendo bien el trabajo, empezando por los indicadores:


**Métricas de evaluación del modelo**:

* **loss**: Compara y mide cuan bueno/malo fue el resultado de la predicción en relación con el resultado correcto. Cuanto más proximo a 0 sea, mejor, queremos que la divergencia entre el valor estima y el esperado sea lo más pequeña posible.

* **mse**: El *error cuadrático medio* (median standard error)es una función de coste. Se pueden utilizar métodos estadísticos formales para determinar la confianza del modelo entrenado. 

* **learning rate**: El valor adecuado de este hiperparámetro depende del problema en cuestión, suele denominarse también *step size*. En general, una buena regla es que si nuestro modelo de aprendizaje no funciona, disminuyamos la learning rate. Si sabemos que el gradiente de la función de loss es pequeño, entonces es seguro probar con learning rate que compensen el gradiente.

* **acurracy**: La exactitud mide el porcentaje de casos en los que el modelo ha acertado y no distingue entre tipos de errores. Es una medida que se debe interpretar con cuidado, ya que puede dar buenos resultados sin un buen modelo cuando las clases están desbalanceadas. En nuestro caso será una de las métricas que usemos dado que no es relevante si tenemos falsos negativos ni falsos positivos.

* **precision**: La precisión mide la **calidad** del modelo. Es el resultado de dividir los verdaderos positivos entre la suma de los verdaderos positivos y los falsos positivos.

* **recall**: La exhaustividad nos aporta información sobre la **cantidad** de elementos que es capaz de identificar. Es el número de resultante de dividir los verdaderos positivos entre la suma de los verdaderos positivos y los falsos negativos.
* **f1**: El Valor-F combina las medidas de precisión y recall en un solo valor, siendo el resultado de multiplicar por dos el producto de la precision y el recall entre la suma de los mismos.


#### Hiperparámetros

Los *hiperparámetros* se utilizan para describir la configuración del modelo. No se utilizan para modelar los datos directamente, pero influyen en la capacidad y características de aprendizaje del modelo. 

Las funciones <code>callback</code> son aquellas que se pasan a otra función como argumento y se ejecutan dentro de esta. Aplicaremos las siguientes:

* **EarlyStopping**: Para cuando la función de coste no mejore en un número dado de epochs. Nos ayudará reduciendo el **overfitting**. Para ello marcaremos <code>verbose</code> en 1, para saber el epoch en el que el modelo se ha parado. Con <code>patience</code> le indicamos cuantos epochs tienen que pasar para que el entrenamiento pare y con <code>min_delta</code> establecemos un incremento específico de la mejora para el error cuadrático.


* **ReduceLROnPlateau**: Si el entrenamiento no mejora tras unos epochs específicos, reduce el valor de learning rate del modelo, lo que normalmente supone una mejora del entrenamiento. Ahora bien, el mejor learning rate  en general es aquel que disminuye a medida que el modelo se acerca a una solución.

* **Batch**: Con el *bach* definimos el número de muestras para trabajar antes de actualizar los paramétros internos del modelo. Las predicciones se comparan con las variables de salidad esperadas y se calcula el error. A partir de este error el algoritmo se actualiza para mejorarse.

    * **Batch Gradient Descent**. Cuando el tamaño del bach es igual que el del conjunto de entrenamiento.
    * **Stochastic Gradient Descent**. Cuando el tamaño del bach es igual a 1.
    * **Mini-Batch Gradient Descent**. Cuando el tamaño del bach está entre uno y el tamaño del conjunto de entrenamient, los más frecuentes en tutoriales son de  32, 64 y 128.
    
    
* **Epoch**: Se trata de un hiperparámetro que define el número de veces que el algoritmo de aprendizaje funcionará sobre el conjunto de datos de entrenamiento. Cada muestra del conjunto de datos de entrenamiento tiene la "oportunidad" de actualizar los parámetros internos del modelo. Puede estar compuesto por uno o más *batches*. El número de *epochs* suele ser grande, lo que permite que el algoritmo se ejecute hasta que el error del modelo se minimice lo suficiente.


#### Red Neuronal Simple

Una red neuronal es un grupo interconectado de nodos de forma similar a las neuronas de un cerebro. Una red neuronal simple se caracteriza por tener un número de entradas y un número de salidas. Cada entrada tendrá un peso e influirá en la salida de la neurona.

##### Creamos la red

Empezaremos definiendo una red neuronal simple y junto con ella, algunos conceptos que se irán repitiendo a medida que vayamos viendo diferentes modelos:
* **Sequential()**: Agrupa de forma lineal las capas del modelo proporcionando características de capacitación e inferencia al modelo.
* **Flatten**: Convierte la matriz de entrada en un array de 1 dimensión (plano).
* **Dense**: Añade una capa oculta a la red neuronal.

**Función de activación ReLu**: Transforma los valores introducidos anulando los valores negativos y dejando los positivos tal y como entran. La ventaja de usar la función ReLU radica en que tiene un comportamiento lineal para entradas positivas, lo que evita precisamente este "estancamiento" durante el entrenamiento. Se activa un solo nodo si la entrada está por encima de cierto umbral.

**Función de salida SoftMax**: Ha resultado dar buenos resultados cuando el entrenamiento es multietiqueta pero no multiclase, tenemos muchos artistas diferentes pero todos son pintores. Asigna probabilidades decimales a cada clase en un caso de clases múltiples de manera que terminen sumando 1. Esta restricción adicional permite que el entrenamiento converja más rápido.

Compilamos el modelo creado y le pasamos los parámetros para la función de pérdida, el optimizador y las métricas a tener en cuenta.
Con <code>.summary()</code> podemos ver un resumen de nuestra red neuronal. Esta red calculo algo más de **3 millones** de parámetros.

##### Entrenar el modelo

Ahora ya solo queda entrenarla, para lo cual le indicamos nuestras imágenes, los pintores y los parámetros que hemos establecido antes. Usaremos la función <code>.fit_generator()</code> en lugar de <code>.fit()</code> dado que nos permite llamar a las características que hemos establecido antes con <code>ImageDataGenerator</code> para aumentar el número de imágenes a analizar con pequeñas modificaciones.

Este modelo ha necesitado 11 *epochs* de los 100 establecidos al principio, al no pararse por la función de <code>EarlyStopping</code>. Podríamos concluir que para llegar a los resultados óptimos deberíamos aumentar este valor. Cada uno ha tardado uno 4 segundos, lo que ha hecho que el modelo tarde algo menos de **1 minuto** en total.

##### Evaluación del modelo

Ahora ya solo queda entrenarla, para lo cual le indicamos nuestras imágenes, los pintores y los parámetros que hemos establecido antes. Usaremos la función <code>.fit_generator()</code> en lugar de <code>.fit()</code> dado que nos permite llamar a las características que hemos establecido antes con <code>ImageDataGenerator</code> para aumentar el número de imágenes a analizar con pequeñas modificaciones.

Este modelo ha necesitado 11 *epochs* de los 100 establecidos al principio, al no pararse por la función de <code>EarlyStopping</code>. Podríamos concluir que para llegar a los resultados óptimos deberíamos aumentar este valor. Cada uno ha tardado uno 4 segundos, lo que ha hecho que el modelo tarde algo menos de **1 minuto** en total.

Con la función creada <code>plot_train_vs_test()</code> podemos ver el comportamiento de las métricas a lo largo de los *epochs*. Podemos observar en la gráfica de *accuracy* como el modelo parece que tiene una tendencia a seguir aumentando este parámetro con un mayor número de *epochs*. Mientras que en el gráfico de la función de pérdida vemos como el valor desciende muy rápido al principio para estancarse a partir de los 11 *epochs*.

<img src="img/snn_evaluation.jpeg" alt="snn_evaluation"/>

##### Predicción del modelo

Una vez entrenado el modelo procedemos a ver los resultados que obtenemos con el conjunto de datos de test. Para ello empezaremos observando la matriz de confusión donde se pueden apreciar las etiquetas reales, eje de las abscisas, frente a las predichas, eje de las ordenadas. Podemos observar que el comportamiento y la capacidad de predicción no es del todo mala.

<img src="img/matriz_snn.jpeg" alt="matriz_snn"/>





* Redes Convolucionales
* VGG-16
* ResNet
* DenseNet121
* NASNet

### Fine Tuning<a name="id10"></a>

### Sistema de recomendación(EXTRA)<a name="id11"></a>

<hr style="color: #7acaff;" width="50%" />
<a name="results"></a>

## Results<a name="id4"></a>



<hr style="color: #7acaff;" width="50%" />
<a name="bibliografia"></a>

## Bibliografía<a name="id5"></a>

* Evans, O. (2019). Sensory Optimization: Neural Networks as a Model for Understanding and Creating Art. Recuperado de https://owainevans.github.io/visual_aesthetics/sensory-optimization.html

* Utrera Brugal, Jesús (2019). Tratamiento de imágenes usando ImageDataGenerator en Keras. Recuperado de https://enmilocalfunciona.io/author/jesus/

<img src="https://www.sketch.ca/sketchPub/uploads/2019/03/radical-art-of-young-people-2000x940.jpg" alt="ART" width="1000" height="50" align="center"/>

<hr style="color: #7acaff;" width="50%" />

<img src="https://bit.ly/2VnXWr2" alt="fondo" width="100" align="center"/>
