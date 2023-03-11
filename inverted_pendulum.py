import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense


#_FUNCIONES GENERALES_****************************************************************************************

# Imprime en la consola una barra de progreso
def progress(text, num, of, width=40):
    percent = int(num * 100 / of)
    left = width * percent // 100
    right = width - left
    print("{}│{}{}│ {}/{} [{}%]".format(text, '█'*left, '┈'*right, num, of, percent), end='\r', flush=True)

# Guarda el modelo del mejor individuo
def save_model(individual):
    individual.save('model_best_individual.h5')


#_FUNCIONES ALGORITMO GENÉTICO_*******************************************************************************

# Genera una población de individuos (redes neuronales) con pesos aleatorios
def generate_population(pop_size):
    input_size    = 4   # Neuronas de entrada
    hidden_size   = 8   # Neuronas de cada capa oculta
    output_size   = 1   # Neuronas de salida
    hidden_layers = 1   # Capas ocultas
    population = []
    for _ in range(pop_size):
        individual = Sequential()
        individual.add(Input(shape=(input_size,))) # Capa de entrada
        for _ in range(hidden_layers):
            individual.add(Dense(hidden_size)) # Capa(s) oculta(s)
        individual.add(Dense(output_size, activation='tanh')) # Capa de salida
        individual.compile()
        population.append(individual)
    return population

# Evalúa el rendimiento de un individuo (red neuronal) en el entorno
def evaluate(individual, environment):
    observation, info = environment.reset(seed=42)
    total_step = 0
    total_fitness = 0
    while True:
        action = individual(observation.reshape(1, -1), training=False).numpy()[0]  # El individuo (red neuronal) predice la acción
        action *= 3
        observation, reward, terminated, truncated, info = environment.step(action)
        position_cart = observation[0]
        vertical_angle_pole = observation[1]
        total_step += 1
        total_fitness += 1 if -0.1 <= position_cart <= 0.1 and -0.2 <= vertical_angle_pole <= 0.2 else 0  # Suma 1 si el carro se encuentra en el centro y el palo vertical
        terminated = False if -0.8 <= vertical_angle_pole <= 0.8 else True
        truncated = total_step == 1000
        if terminated or truncated: # Termina la evaluación del individuo
            break
    return total_fitness

# Selecciona los mejores individuos para reproducirse
def selection(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        index_parent_max = np.argmax(fitnesses)
        parents.append(population[index_parent_max])
        del population[index_parent_max]
        del fitnesses[index_parent_max]
    return parents

# Cruza dos individuos (redes neuronales) para producir un descendiente
def crossover(parent1, parent2):
    # Obtiene los pesos (genes) de todas las capas (ocultas + salida) de cada padre
    weights1 = [layer.get_weights() for layer in parent1.layers]
    weights2 = [layer.get_weights() for layer in parent2.layers]

    # Crea el descendiente
    child = Sequential()
    child.add(Input(shape=(parent1.layers[0].input_shape[1],)))
    layers_size = len(parent1.layers)  # Cantidad de capas (ocultas + salida)
    for i in range(layers_size):
        activation_function = 'tanh' if i == layers_size-1 else None
        child.add(Dense(parent1.layers[i].output_shape[1], activation=activation_function, weights=[
            np.where(np.random.random(weights1[i][0].shape) < 0.5, weights1[i][0], weights2[i][0]),
            np.where(np.random.random(weights1[i][1].shape) < 0.5, weights1[i][1], weights2[i][1])
        ]))
    child.compile()
    return child

# Muta un individuo (red neuronal) con una cierta probabilidad
def mutate(individual, mutation_rate):
    for layer in individual.layers:
        weights = layer.get_weights()
        for i in range(len(weights[0])):
            for j in range(len(weights[0][0])):
                if np.random.uniform(0, 1) < mutation_rate:
                    weights[0][i][j] = np.random.uniform(-1, 1)
        layer.set_weights(weights)


#_ALGORITMO GENÉTICO_*****************************************************************************************

def genetic_algorithm(pop_size, num_parents, num_generations, mutation_rate, environment):
    # Generamos la población inicial
    population = generate_population(pop_size)

    # Iteramos durante el número de generaciones especificado
    for generation in range(num_generations):
        # En la primera generación este código se omite
        if generation >= 1:
            # Seleccionamos los mejores individuos para reproducirse
            parents = selection(population, fitnesses, num_parents)

            # Generamos la siguiente generación de individuos
            next_generation = []
            #next_generation.append(parents[0])
            for _ in range(pop_size):
                # Seleccionamos dos padres al azar
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)

                # Cruzamos los padres para producir un descendiente
                child = crossover(parent1, parent2)

                # Mutamos al descendiente con una cierta probabilidad
                mutate(child, mutation_rate)

                # Añadimos el descendiente a la siguiente generación
                next_generation.append(child)

            # Actualizamos la población actual
            population = next_generation

        # Inicio
        print("Generation: {}".format(generation+1))

        # Evaluamos a cada individuo de la población y guardamos sus fitnesses
        fitnesses = []
        for individual in population:
            progress(text='Training: ', num=len(fitnesses)+1, of=pop_size) # Impriminos el progreso del entrenamiento para una generación
            fitness = evaluate(individual, environment)
            fitnesses.append(fitness)
        
        # Imprimimos los fitnesses de todos los individuos (ordenados de mayor a menor) y el fitness promedio de la población
        print("Fitnesses: {} - Average: {:.2f}\n".format(sorted(fitnesses, reverse=True), np.mean(fitnesses)))

        # Terminamos el entrenamiento si algún individuo obtiene la máxima puntuación (1000 = aprendió)
        if 1000 in fitnesses:
            break

    # Devolvemos el mejor individuo de la última generación
    return population[np.argmax(fitnesses)]


#_MAIN_*******************************************************************************************************

# Definimos los parámetros del algoritmo genético
pop_size = 20           # Tamaño de la población
num_parents = 5         # Número de padres
num_generations = 20    # Número de generaciones
mutation_rate = 0.05    # Probabilidad de que ocurra una mutación en cada gen (0.05 = 5%)

# Definimos los parámetros del entorno
env_name = "InvertedPendulum-v4"  # Nombre del entorno
env_render_mode = "human"         # Renderizado del entorno (None = No se renderiza / "human" = Si se renderiza)

# Creamos el entorno
environment = gym.make(env_name, render_mode=env_render_mode)

# Ejecutamos el algoritmo genético y obtenemos el mejor individuo
best_individual = genetic_algorithm(pop_size, num_parents, num_generations, mutation_rate, environment)

# Evaluamos el rendimiento del mejor individuo
best_fitness = evaluate(best_individual, environment)

# Cerramos el entorno
environment.close()

# Guardamos el modelo
#save_model(best_individual)

print('Training completed!')
print('Name of the environment: "{}"'.format(env_name))
print('Fitness of the best individual: {}'.format(best_fitness))