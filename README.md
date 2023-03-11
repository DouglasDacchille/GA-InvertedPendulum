# GA-InvertedPendulum

En este proyecto, se emplea el entrenamiento de redes neuronales mediante algoritmos genéticos para resolver el problema del péndulo invertido de OpenAI Gym.

## Tabla de contenido

* [Descripción](#descripción)
* [Requerimientos](#requerimientos)
* [Instalación](#instalación)
* [Como usar](#como-usar)
* [Logs de ejecución](#logs-de-ejecución)
* [Resultados](#resultados)
* [Conclusión](#conclusión)

## Descripción

El péndulo invertido es un problema clásico de control que consta de un carro que puede moverse linealmente, sobre el cual se fija el extremo de un poste vertical. El carro puede empujarse hacia la izquierda o derecha, y el objetivo es equilibrar el poste en la parte superior del carro aplicando fuerzas sobre éste.

| Péndulo invertido                                                                                    |
| ---------------------------------------------------------------------------------------------------- |
| ![image_01](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_01.png) |

<br>

Se utilizó la biblioteca `'InvertedPendulum-v4'` de OpenAI Gym, la cual simula un péndulo invertido y proporciona un entorno de aprendizaje para agentes de control.

| Simulación del péndulo invertido                                                                     |
| ---------------------------------------------------------------------------------------------------- |
| ![image_02](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_02.png) |

<br>

Dicha biblioteca, permite al agente (controlador) interactuar con el entorno (péndulo invertido) y recibir información sobre su estado (posición del carro, ángulo del poste, velocidad del carro y velocidad angular del poste) mediante la observación. El agente envía una acción (fuerza aplicada al carro) para mantener el péndulo en posición vertical. La calidad de la acción tomada por el agente se mide a través de una recompensa que indica el desempeño del agente en el entorno.

| Interacción del agente con el entorno                                                                |
| ---------------------------------------------------------------------------------------------------- |
| ![image_03](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_03.png) |

<br>

Para representar al agente, se utilizó una red neuronal sencilla de cuatro neuronas de entrada, una capa oculta de ocho neuronas y una neurona de salida.

El objetivo del agente es aprender a equilibrar el péndulo invertido de manera sostenible, incluso en presencia de perturbaciones externas. Para lograrlo, la red neuronal fue entrenada utilizando algoritmos genéticos.

El proceso de entrenamiento comienza con una población inicial de individuos, cada uno representado por una red neuronal con una combinación aleatoria de parámetros. En cada generación, se evalúa el rendimiento de cada individuo y se seleccionan los más aptos para reproducirse.

Durante la reproducción, se combinan los parámetros de los individuos seleccionados y se introducen mutaciones aleatorias, lo que da lugar a nuevos individuos que heredan las características más beneficiosas de sus progenitores.

Este proceso de selección y reproducción se repite durante varias generaciones, permitiendo que la población evolucione y mejore gradualmente hasta obtener un individuo capaz de realizar la tarea de manera satisfactoria.

| Algoritmo Genético                                                                                   |
| ---------------------------------------------------------------------------------------------------- |
| ![image_04](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_04.png) |

## Requerimientos

* python 3.10.8
* gym 0.26.2
* keras 2.11.0
* numpy 1.23.5
* tensorflow 2.11.0

## Instalación

1. Clonar el repositorio:

```bash
$ git clone https://github.com/DouglasDacchille/GA-InvertedPendulum
```

2. Instalar los paquetes necesarios:

```shell
$ pip install -r requirements.txt
```
    
## Como usar

1. Abrir el archivo `inverted_pendulum.py`.

2. Ajustar los parámetros del algoritmo genético:

```python
154   # Definimos los parámetros del algoritmo genético
155   pop_size = 20           # Tamaño de la población
156   num_parents = 5         # Número de padres
157   num_generations = 20    # Número de generaciones
158   mutation_rate = 0.05    # Probabilidad de que ocurra una mutación en cada gen (0.05 = 5%)
```

3. Ajustar los parámetros del entorno:

```python
160   # Definimos los parámetros del entorno
161   env_name = "InvertedPendulum-v4"  # Nombre del entorno
162   env_render_mode = "human"         # Renderizado del entorno (None = No se renderiza / "human" = Si se renderiza)
```

4. Ajustar los parámetros de los individuos (redes neuronales):

```python
23  # Genera una población de individuos (redes neuronales) con pesos aleatorios
24   def generate_population(pop_size):
25       input_size    = 4   # Neuronas de entrada
26       hidden_size   = 8   # Neuronas de cada capa oculta
27       output_size   = 1   # Neuronas de salida
28       hidden_layers = 1   # Capas ocultas
```

5. Ejecutar el archivo `inverted_pendulum.py` utilizando el siguiente comando:

```python
$ python inverted_pendulum.py
```

## Logs de ejecución

* Durante la ejecución, se observará una barra de progreso que indicará el avance en tiempo real del entrenamiento de los individuos en cada generación:

```shell
Generation: 1
Training: │██████████████████████████┈┈┈┈┈┈┈┈┈┈┈┈┈┈│ 13/20 [65%]
```

* Tras finalizar el entrenamiento de cada generación, se imprimirá la puntuación de cada individuo (ordenado de mayor a menor) y el promedio de la población:

```shell
Generation: 1
Fitnesses: [50, 48, 41, 40, 38, 38, 25, 25, 25, 18, 17, 15, 15, 13, 11, 10, 9, 6, 6, 6] - Average: 22.80
```

* Una vez finalizado el entrenamiento, se generará un archivo llamado `best_individual_model.h5`, que contendrá el modelo de red neuronal del mejor individuo. Por defecto, esta instrucción se encuentra deshabilitada:

```python
176   # Guardamos el modelo
177   #save_model(best_individual)
```

## Resultados

* Prueba con población inicial:

Al ejecutar el archivo `inverted_pendulum.py`, se crea una población inicial de individuos con genes aleatorios, lo que implica un bajo rendimiento general de la población en las primeras iteraciones. A continuación, se muestra un video que ilustra el comportamiento del péndulo durante la primera generación:

| ![image_05](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_05.gif) |
| ---------------------------------------------------------------------------------------------------- |

```shell
Generation: 1
Fitnesses: [86, 55, 54, 47, 44, 37, 26, 18, 18, 18, 16, 14, 13, 12, 9, 7, 6, 6, 6, 5] - Average: 24.85
```

<br>

* Prueba del mejor individuo:

Después de varias generaciones (9 en este caso), se logró evolucionar una red neuronal capaz de equilibrar el péndulo invertido de manera sostenible y con alta precisión, incluso en presencia de perturbaciones externas.

En el video se observa el momento exacto cuando se aplican dichas perturbaciones (flechas).

| ![image_06](https://github.com/DouglasDacchille/GA-InvertedPendulum/blob/master/images/image_06.gif) |
| ---------------------------------------------------------------------------------------------------- |

```shell
Generation: 1
Fitnesses: [86, 55, 54, 47, 44, 37, 26, 18, 18, 18, 16, 14, 13, 12, 9, 7, 6, 6, 6, 5] - Average: 24.85

Generation: 2
Fitnesses: [143, 102, 73, 67, 49, 49, 44, 40, 39, 29, 23, 21, 19, 14, 12, 12, 9, 9, 8, 8] - Average: 38.50

Generation: 3
Fitnesses: [204, 196, 131, 108, 89, 65, 52, 47, 46, 45, 43, 40, 29, 23, 21, 14, 10, 8, 7, 5] - Average: 59.15

Generation: 4
Fitnesses: [293, 243, 187, 179, 176, 131, 123, 113, 68, 61, 61, 51, 49, 44, 42, 38, 32, 23, 11, 8] - Average: 96.65

Generation: 5
Fitnesses: [450, 343, 312, 253, 211, 198, 170, 155, 154, 129, 116, 80, 68, 63, 62, 58, 33, 18, 6, 6] - Average: 144.25

Generation: 6
Fitnesses: [626, 327, 296, 271, 262, 186, 156, 135, 129, 113, 111, 94, 77, 66, 52, 44, 34, 25, 18, 11] - Average: 151.65

Generation: 7
Fitnesses: [736, 620, 301, 247, 194, 174, 168, 151, 149, 147, 117, 92, 53, 53, 51, 44, 39, 27, 22, 9] - Average: 169.70

Generation: 8
Fitnesses: [953, 670, 643, 593, 582, 547, 439, 383, 340, 319, 313, 284, 198, 186, 162, 146, 127, 86, 50, 42] - Average: 353.15

Generation: 9
Fitnesses: [1000, 953, 682, 588, 552, 438, 403, 381, 375, 321, 242, 229, 224, 214, 187, 124, 119, 112, 109, 84] - Average: 366.85

Training completed!
Name of the environment: "InvertedPendulum-v4"
Fitness of the best individual: 1000
```

## Conclusión

Los resultados obtenidos demuestran la eficacia de la combinación de redes neuronales y algoritmos genéticos para la resolución de problemas de control de este tipo. Se sugiere su posible aplicación en otros contextos.