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

Crear una red neuronal convolucional para reconocer a los artistas y su movimiento. Como EXTRA me gustaría poder crear también un sistema de recomendación de forma que puedas obtener como output no solo el artista y el movimiento al que pertenece, también una selección de cuadros similares.

<hr style="color: #7acaff;" width="50%" />
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

### EDA<a name="id7"></a>

<img src="img/grafo.png" alt="Grafo"/>

<img src="img/random_pictures.png" alt="Random"/>


### Data Wrangling<a name="id8"></a>

### Choosing a Model<a name="id9"></a>

### Fine Tuning<a name="id10"></a>

### Sistema de recomendación(EXTRA)<a name="id11"></a>

<hr style="color: #7acaff;" width="50%" />
<a name="results"></a>

## Results<a name="id4"></a>



<hr style="color: #7acaff;" width="50%" />
<a name="bibliografia"></a>

## Bibliografía<a name="id5"></a>

* Evans, O. (2019). Sensory Optimization: Neural Networks as a Model for Understanding and Creating Art. Recuperado de https://owainevans.github.io/visual_aesthetics/sensory-optimization.html

<img src="https://www.sketch.ca/sketchPub/uploads/2019/03/radical-art-of-young-people-2000x940.jpg" alt="ART" width="1000" height="50" align="center"/>

<hr style="color: #7acaff;" width="50%" />

<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100" align="center"/>