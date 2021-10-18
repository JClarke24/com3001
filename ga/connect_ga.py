import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from game import Game
from player import Player
from gameController import GameController
from model import ConnectFourModel

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_DRAW = 0

number_of_neurons = 42 + 42 + 3
number_of_neurons_l1 = 42
number_of_neurons_l2 = 42
number_of_neurons_l3 = 3
number_of_weights = 42 * 42 + 42 * 42 + 42 * 3
number_of_weights_l1 = 42 * 42
number_of_weights_l2 = 42 * 42
number_of_weights_l3 = 42 * 3


class ConnectGa:

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

        model = Sequential()
        model.add(Dense(number_of_neurons_l1, activation='relu', input_shape=(number_of_neurons_l1,)))
        model.add(Dense(number_of_neurons_l2, activation='relu'))
        model.add(Dense(number_of_neurons_l3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

        model.layers[0].set_weights([layer1, bias1])
        model.layers[1].set_weights([layer2, bias2])
        model.layers[2].set_weights([layer3, bias3])

        game1 = Game()
        game2 = Game()
        redPlayer = Player(RED_PLAYER_VAL, strategy='random')
        yellowPlayer = Player(YELLOW_PLAYER_VAL, strategy='random')
        redGaPlayer = Player(RED_PLAYER_VAL, strategy='model', model=model)
        yellowGaPlayer = Player(YELLOW_PLAYER_VAL, strategy='model', model=model)

        gameController1 = GameController(game1, redGaPlayer, yellowPlayer)
        gameController2 = GameController(game2, redPlayer, yellowGaPlayer)

        gameController1.playGame()
        gameController2.playGame()

        score1 = 0

        if game1.getGameResult() == RED_PLAYER_VAL:
            score1 = 200
        elif game1.getGameResult() == GAME_STATE_DRAW:
            score1 = 46
        else:
            score1 = len(game1.getBoardHistory())

        score2 = 0

        if game2.getGameResult() == RED_PLAYER_VAL:
            score2 = 200
        elif game2.getGameResult() == GAME_STATE_DRAW:
            score2 = 46
        else:
            score2 = len(game2.getBoardHistory())

        return (score1 + score2) / 2
