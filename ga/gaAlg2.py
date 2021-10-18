import numpy
import GA
import ANN
import matplotlib.pyplot
from tensorflow import keras

from game import Game
from player import Player
from gameController import GameController
from model import ConnectFourModel

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_NOT_ENDED = 2

firstGame = Game()

model = keras.models.load_model('C:/Users/User/OneDrive - University of Surrey/Year 3/COM3001 Professional Project/connect4-nn/models')
nnModel = ConnectFourModel(42, 3, 50, 100, model=model)
redNeuralPlayer = Player(RED_PLAYER_VAL, strategy='model', model=nnModel)
yellowNeuralPlayer = Player(YELLOW_PLAYER_VAL, strategy='model', model=nnModel)

gameController = GameController(firstGame, redNeuralPlayer, yellowNeuralPlayer)
print ("Playing with both neural network players")
gameController.simulateManyGames(1000)

trainingHist = gameController.getTrainingHistory()

data_inputs = []
data_outputs = []

for hist in trainingHist:
    data_inputs.append(hist[1])
    data_outputs.append(hist[0])

data_inputs = numpy.asarray(data_inputs)
data_inputs = numpy.reshape(data_inputs, newshape=(data_inputs.shape[0], data_inputs.shape[1]*data_inputs.shape[2]))
data_outputs = numpy.asarray(data_outputs)

print("input shape: ", data_inputs.shape)
print("output shape: ", data_outputs.shape)

sol_per_pop = 8
num_parents_mating = 4
num_generations = 50
mutation_percent = 10

initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
    HL1_neurons = 42
    input_HL1_weights = numpy.random.uniform(low=-1, high=1, size=(data_inputs.shape[1], HL1_neurons))

    HL2_neurons = 42
    HL1_HL2_weights = numpy.random.uniform(low=-1, high=1, size=(HL1_neurons, HL2_neurons))

    output_neurons = 3
    HL2_output_weights = numpy.random.uniform(low=-1, high=1, size=(HL2_neurons, output_neurons))

    model_weights = [input_HL1_weights, HL1_HL2_weights, HL2_output_weights]
    initial_pop_weights.append(model_weights)

pop_weights_mat = initial_pop_weights
pop_weights_vector = GA.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = numpy.empty(shape=num_generations)

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(pop_weights_mat,
                          data_inputs,
                          data_outputs,
                          activation="relu")

    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(pop_weights_vector,
                                    fitness.copy(),
                                    num_parents_mating)
    print("Parents")
    #print(parents)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    print("Crossover")
    #print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover,
                                     mutation_percent=mutation_percent)
    print("Mutation")
    #print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="relu")
print("Accuracy of the best solution is : ", acc)

matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=15)
