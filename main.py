from game import Game
from player import  Player
from gameController import GameController
from model import ConnectFourModel

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
GAME_STATE_NOT_ENDED = 2

if __name__ == "__main__":
    firstGame = Game()
    redPlayer = Player(RED_PLAYER_VAL, strategy='random')
    yellowPlayer = Player(YELLOW_PLAYER_VAL, strategy='random')

    gameController = GameController(firstGame, redPlayer, yellowPlayer)
    print ("Playing with both players with random strategies")
    gameController.simulateManyGames(10000)

    model = ConnectFourModel(42, 3, 50, 100)
    model.train(gameController.getTrainingHistory())
    model.model.save('C:/Users/User/OneDrive - University of Surrey/Year 3/COM3001 Professional Project/connect4-nn/models')

    redNeuralPlayer = Player(RED_PLAYER_VAL, strategy='model', model=model)
    yellowNeuralPlayer = Player(YELLOW_PLAYER_VAL, strategy='model', model=model)

    secondGame = Game()
    gameController = GameController(secondGame, redPlayer, yellowNeuralPlayer)
    print ("Playing with yellow player as Neural Network")
    gameController.simulateManyGames(10)

    thirdGame = Game()
    gameController = GameController(thirdGame, redNeuralPlayer, yellowPlayer)
    print("Playing with red player as Neural Network")
    gameController.simulateManyGames(10)
