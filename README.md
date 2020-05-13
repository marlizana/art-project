<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100" align="center"/>

<img src="https://www.sketch.ca/sketchPub/uploads/2019/03/radical-art-of-young-people-2000x940.jpg" alt="ART" width="1000" align="center"/>

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


<a name="project"></a>

## Project Description<a name="id1"></a>

Crear una red neuronal convolucional para reconocer a los artistas y su movimiento. Como EXTRA me gustaría poder crear también un sistema de recomendación de forma que puedas obtener como output no solo el artista y el movimiento al que pertenece, también una selección de cuadros similares.

<a name="dataset"></a>

## Dataset<a name="id2"></a>

Hemos seleccionado un dataset de la plataforma Kaggle sobre <a href="https://www.kaggle.com/dannielr/marvel-superheroes/">los 50 artistas más influyentes</a> de la historia del arte. Se trata de una colección de obras de arte y un conjunto de datos con información básica recuperada de wikipedia en formato CSV. 

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

<a name="workflow"></a>

## Workflow<a name="id3"></a>

El recorrido del proyecto es el siguiente:

1. Análisis del proyecto
2. EDA
3. Data Wrangling
4. Choosing a Model
5. Fine Tuning
6. Sistema de recomendación(EXTRA)

<a name="results"></a>

## Results<a name="id4"></a>



## Bibliografía<a name="id5"></a>

* Evans, O. (2019). Sensory Optimization: Neural Networks as a Model for Understanding and Creating Art. Recuperado de https://owainevans.github.io/visual_aesthetics/sensory-optimization.html