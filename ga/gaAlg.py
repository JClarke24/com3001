import numpy
import pygad
from tensorflow import keras

from game import Game
from player import Player
from gameController import GameController
from model import ConnectFourModel

num_generations = 20
num_parents_mating = 4

# fitness_function = fitness_func

sol_per_pop = 8
num_genes = (42 * 42) + 42 + (42 * 42) + 42 + (42 * 3) + 3

init_range_low = -1
init_range_high = 1

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_DRAW = 0

number_of_neurons = 42 + 42 + 42 + 3
number_of_neurons_l1 = 42
number_of_neurons_l2 = 42
number_of_neurons_l3 = 3
number_of_weights = 42 * 42 + 42 * 42 + 42 * 3
number_of_weights_l1 = 42 * 42
number_of_weights_l2 = 42 * 42
number_of_weights_l3 = 42 * 3

# model = Sequential()
    # model.add(Dense(number_of_neurons_l1, activation='relu', input_shape=(number_of_neurons_l1,)))
    # model.add(Dense(number_of_neurons_l2, activation='relu'))
    # model.add(Dense(number_of_neurons_l3, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
model = keras.models.load_model(
        'C:/Users/User/OneDrive - University of Surrey/Year 3/COM3001 Professional Project/connect4-nn/models')
nnModel = ConnectFourModel(number_of_neurons_l1, number_of_neurons_l3, 50, 100, model=model)


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


def fitness_func(solution, solution_idx):
    layer1 = solution[0:number_of_weights_l1]
    bias1 = solution[number_of_weights_l1:number_of_weights_l1 + number_of_neurons_l1]
    layer2 = solution[
             number_of_weights_l1 + number_of_neurons_l1:number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2]
    bias2 = solution[
            number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2:number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2 + number_of_neurons_l2]
    layer3 = solution[
             number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2 + number_of_neurons_l2:number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2 + number_of_neurons_l2 + number_of_weights_l3]
    bias3 = solution[
            number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2 + number_of_neurons_l2 + number_of_weights_l3:number_of_weights_l1 + number_of_neurons_l1 + number_of_weights_l2 + number_of_neurons_l2 + number_of_weights_l3 + number_of_neurons_l3]
    layer1 = layer1.reshape((42, 42))
    layer2 = layer2.reshape((42, 42))
    layer3 = layer3.reshape((42, 3))

    model.layers[0].set_weights([layer1, bias1])
    model.layers[1].set_weights([layer2, bias2])
    model.layers[2].set_weights([layer3, bias3])

    gaModel = ConnectFourModel(number_of_neurons_l1, number_of_neurons_l3, 50, 100, model=model)

    game1 = Game()
    game2 = Game()
    redPlayer = Player(RED_PLAYER_VAL, strategy='model', model=nnModel)
    yellowPlayer = Player(YELLOW_PLAYER_VAL, strategy='model', model=nnModel)
    redGaPlayer = Player(RED_PLAYER_VAL, strategy='model', model=gaModel)
    yellowGaPlayer = Player(YELLOW_PLAYER_VAL, strategy='model', model=gaModel)

    gameController1 = GameController(game1, redGaPlayer, yellowPlayer)
    gameController2 = GameController(game2, redPlayer, yellowGaPlayer)

    gameController1.playGame()
    gameController2.playGame()

    score1 = 0

    if game1.getGameResult() == RED_PLAYER_VAL:
        score1 += 100
    elif game1.getGameResult() == GAME_STATE_DRAW:
        score1 += 42
    else:
        score1 += len(game1.getBoardHistory())/2

    score2 = 0

    if game2.getGameResult() == YELLOW_PLAYER_VAL:
        score2 += 100
    elif game2.getGameResult() == GAME_STATE_DRAW:
        score2 += 42
    else:
        score2 += len(game2.getBoardHistory())/2

    return (score1 + score2) / 2


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_gen)

ga_instance.run()

ga_instance.plot_result()
