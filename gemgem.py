# Gemgem (a Bejeweled clone)
# By Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

"""
This program has "gem data structures", which are basically dictionaries
with the following keys:
  'x' and 'y' - The location of the gem on the board. 0,0 is the top left.
                There is also a ROWABOVEBOARD row that 'y' can be set to,
                to indicate that it is above the board.
  'direction' - one of the four constant variables UP, DOWN, LEFT, RIGHT.
                This is the direction the gem is moving.
  'imageNum'  - The integer index into GEMLABELS to denote which image
                this gem uses.
"""

import random, time, sys, copy
from pprint import pprint

BOARDWIDTH = 8 # how many columns in the board
BOARDHEIGHT = 8 # how many rows in the board

DEDUCTSPEED = 0.8 # reduces score by 1 point every DEDUCTSPEED seconds.
GEMLABELS = [0,1,2,3,4,5]

NUMGEMTYPES = len(GEMLABELS)
assert NUMGEMTYPES >= 5 # game needs at least 5 types of gems to work

MAX_ITERS = 100

# constants for direction values
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

EMPTY_SPACE = -1 # an arbitrary, nonpositive value
ROWABOVEBOARD = 'row above board' # an arbitrary, noninteger value

def main():
    m = Mediator()
    m.runExperiment()

class Critic(object):
    @classmethod
    def genTraining(cls, gameTrace, score, weights):
        # for each game state, find V_train
        training = []
        for i in range(len(gameTrace) - 1):
            state = gameTrace[i]
            successor = gameTrace[i + 1]

            vtrain = targetfunc(successor, weights)
            training.append([state, vtrain])
        # add final state and corresponding score
        #training.append([gameTrace[-1], score)
        return training

class Generalizer(object):
    # learning rate
    eta = 0.1
    @classmethod
    def updateHypothesis(cls, weights, trainingSet):
        newWeights = [None] * len(weights)
        for board, vtrain in trainingSet:
            for i, w in enumerate(weights):
                newWeights[i] = w + eta * (vtrain - targetfunc(board, weights)) * extractFeatures(board) * i
        return newWeights

def extractFeatures(board):
    # count 3/4/5-in-a-rows on the board
    gemsToRemove = findMatchingGems(board)
    num3 = len(filter(lambda x: len(x) == 3, gemsToRemove))
    num4 = len(filter(lambda x: len(x) == 4, gemsToRemove))
    num5 = len(filter(lambda x: len(x) == 5, gemsToRemove))
    return [1, num3, num4, num5]

def targetfunc(state, weights):
    features = extractFeatures(state)
    return sum((features[i] * weights[i] for i in range(len(weights))))

class Mediator(object):
    def __init__(self):
        self.player = Player()
    def runExperiment(self):
        for i in range(10):
            board = Board()
            self.runEpisode(board, self.player)

            gameTrace = board.getTrace()

            trainingSet = Critic.genTraining(gameTrace, board.score, self.player.weights)
            print 'weights:', weights
            weights = Generalizer.updateHypothesis(self.player.weights, trainingSet)
            print 'weights:', weights
            self.player.weights = weights

    def runEpisode(self, board, player):
        while not board.gameover:
            state = board.getObs()
            action = player.getAction(state)

            #print 'score:', board.score
            #pprint(board.getObs())
            #print 'swap:'
            #row1 = int(raw_input('1st row:'))
            #col1 = int(raw_input('1st col:'))
            #row2 = int(raw_input('2nd row:'))
            #col2 = int(raw_input('2nd col:'))
            #action = [{'x': row1, 'y':col1}, {'x': row2, 'y': col2}]

            reward = board.doAction(action)
            player.getReward(reward)
        

# Player is the agent/performance system
class Player(object):
    def __init__(self):
        self.weights = [1, 1, 1, 1]
    def getAction(self, state):
        actions = self.genActions(state)
        return self.chooseAction(state, actions)
    def genActions(self, state):
        # return all possible moves that result in a match
        return possibleMoves(state)
    def chooseAction(self, state, actions):
        values = []
        for a in actions:
            state = swap(state, a)
            values.append({'value':targetfunc(state, self.weights), 'action':a})
            state = swap(state, a)
        # return the move that gives board with the largest targetfunc value
        return max(values, key=lambda x: x['value'])['action']
    def getReward(self, reward):
        pass

def swap(board, pair):
    return board

# Board is the environment (used by performance system)
class Board(object):
    def __init__(self):
        # initalize the board
        self.board = getBlankBoard()
        self.score = 0
        self.trace = []
        fillBoard(self.board) # Drop the initial gems.

        # initialize variables for the start of a new game
        firstSelectedGem = None
        self.gameover = False
    def getObs(self):
        return self.board
    def getTrace():
        return self.trace
    def doAction(self, action):
        scoreAdd = 0
        print action
        firstSelectedGem = {'x': action[0][0], 'y': action[0][1]}
        clickedSpace = {'x': action[1][0], 'y': action[1][1]}
        # Two gems have been clicked on and selected. Swap the gems.
        firstSwappingGem, secondSwappingGem = getSwappingGems(self.board, firstSelectedGem, clickedSpace)
        if firstSwappingGem == None and secondSwappingGem == None:
            # If both are None, then the gems were not adjacent
            print 'gems not adjacent'
            firstSelectedGem = None # deselect the first gem
            return 0

        # Swap the gems in the board data structure.
        self.board[firstSwappingGem['x']][firstSwappingGem['y']] = secondSwappingGem['imageNum']
        self.board[secondSwappingGem['x']][secondSwappingGem['y']] = firstSwappingGem['imageNum']

        # See if this is a matching move.
        matchedGems = findMatchingGems(self.board)
        if matchedGems == []:
            print 'did not cause a match'
            # Was not a matching move; swap the gems back
            self.board[firstSwappingGem['x']][firstSwappingGem['y']] = firstSwappingGem['imageNum']
            self.board[secondSwappingGem['x']][secondSwappingGem['y']] = secondSwappingGem['imageNum']
        else:
            # This was a matching move.
            self.trace.append(self.board)
            while matchedGems != []:
                # Remove matched gems, then pull down the board.

                for gemSet in matchedGems:
                    scoreAdd += (10 + (len(gemSet) - 3) * 10)
                    for gem in gemSet:
                        self.board[gem[0]][gem[1]] = EMPTY_SPACE
                print 'matched! you get points:', scoreAdd
                pprint(self.board)
                self.score += scoreAdd

                # Drop the new gems.
                fillBoard(self.board)

                # Check if there are any new matches.
                matchedGems = findMatchingGems(self.board)
        firstSelectedGem = None

        if not canMakeMove(self.board) or len(self.trace) > MAX_ITERS:
            self.gameover = True
        return scoreAdd

def getSwappingGems(board, firstXY, secondXY):
    # If the gems at the (X, Y) coordinates of the two gems are adjacent,
    # then their 'direction' keys are set to the appropriate direction
    # value to be swapped with each other.
    # Otherwise, (None, None) is returned.
    firstGem = {'imageNum': board[firstXY['x']][firstXY['y']],
                'x': firstXY['x'],
                'y': firstXY['y']}
    secondGem = {'imageNum': board[secondXY['x']][secondXY['y']],
                 'x': secondXY['x'],
                 'y': secondXY['y']}
    highlightedGem = None
    if firstGem['x'] == secondGem['x'] + 1 and firstGem['y'] == secondGem['y']:
        firstGem['direction'] = LEFT
        secondGem['direction'] = RIGHT
    elif firstGem['x'] == secondGem['x'] - 1 and firstGem['y'] == secondGem['y']:
        firstGem['direction'] = RIGHT
        secondGem['direction'] = LEFT
    elif firstGem['y'] == secondGem['y'] + 1 and firstGem['x'] == secondGem['x']:
        firstGem['direction'] = UP
        secondGem['direction'] = DOWN
    elif firstGem['y'] == secondGem['y'] - 1 and firstGem['x'] == secondGem['x']:
        firstGem['direction'] = DOWN
        secondGem['direction'] = UP
    else:
        # These gems are not adjacent and can't be swapped.
        return None, None
    return firstGem, secondGem


def getBlankBoard():
    # Create and return a blank board data structure.
    board = []
    for x in range(BOARDWIDTH):
        board.append([EMPTY_SPACE] * BOARDHEIGHT)
    return board


def canMakeMove(board):
    return len(possibleMoves(board)) > 0
def possibleMoves(board):
    # Return True if the board is in a state where a matching
    # move can be made on it. Otherwise return False.

    # The patterns in oneOffPatterns represent gems that are configured
    # in a way where it only takes one move to make a triplet.
    oneOffPatterns = (((0,1), (1,0), (2,0),    ((0,0), (0,1))),
                      ((0,1), (1,1), (2,0),    ((2,0), (2,1))),
                      ((0,0), (1,1), (2,0),    ((1,0), (1,1))),
                      ((0,1), (1,0), (2,1),    ((1,0), (1,1))),
                      ((0,0), (1,0), (2,1),    ((2,0), (2,1))),
                      ((0,0), (1,1), (2,1),    ((0,0), (0,1))),
                      ((0,0), (0,2), (0,3),    ((0,0), (1,0))),
                      ((0,0), (0,1), (0,3),    ((2,0), (3,0))))

    # The x and y variables iterate over each space on the board.
    # If we use + to represent the currently iterated space on the
    # board, then this pattern: ((0,1), (1,0), (2,0))refers to identical
    # gems being set up like this:
    #
    #     +A
    #     B
    #     C
    #
    # That is, gem A is offset from the + by (0,1), gem B is offset
    # by (1,0), and gem C is offset by (2,0). In this case, gem A can
    # be swapped to the left to form a vertical three-in-a-row triplet.
    #
    # There are eight possible ways for the gems to be one move
    # away from forming a triple, hence oneOffPattern has 8 patterns.

    moves = []
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            for pat in oneOffPatterns:
                # check each possible pattern of "match in next move" to
                # see if a possible move can be made.
                if (getGemAt(board, x+pat[0][0], y+pat[0][1]) == \
                    getGemAt(board, x+pat[1][0], y+pat[1][1]) == \
                    getGemAt(board, x+pat[2][0], y+pat[2][1]) != None):
                    moves.append(map(lambda z: (z[0] + x, z[1] + y), pat[3]))
                if (getGemAt(board, x+pat[0][1], y+pat[0][0]) == \
                    getGemAt(board, x+pat[1][1], y+pat[1][0]) == \
                    getGemAt(board, x+pat[2][1], y+pat[2][0]) != None):
                    moves.append(map(lambda z: (z[1] + x, z[0] + y), pat[3]))
    return moves


def pullDownAllGems(board):
    # pulls down gems on the board to the bottom to fill in any gaps
    for x in range(BOARDWIDTH):
        gemsInColumn = []
        for y in range(BOARDHEIGHT):
            if board[x][y] != EMPTY_SPACE:
                gemsInColumn.append(board[x][y])
        board[x] = ([EMPTY_SPACE] * (BOARDHEIGHT - len(gemsInColumn))) + gemsInColumn


def getGemAt(board, x, y):
    if x < 0 or y < 0 or x >= BOARDWIDTH or y >= BOARDHEIGHT:
        return None
    else:
        return board[x][y]


def getDropSlots(board):
    # Creates a "drop slot" for each column and fills the slot with a
    # number of gems that that column is lacking. This function assumes
    # that the gems have been gravity dropped already.
    boardCopy = copy.deepcopy(board)
    pullDownAllGems(boardCopy)

    dropSlots = []
    for i in range(BOARDWIDTH):
        dropSlots.append([])

    # count the number of empty spaces in each column on the board
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT-1, -1, -1): # start from bottom, going up
            if boardCopy[x][y] == EMPTY_SPACE:
                possibleGems = list(range(len(GEMLABELS)))
                for offsetX, offsetY in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    # Narrow down the possible gems we should put in the
                    # blank space so we don't end up putting an two of
                    # the same gems next to each other when they drop.
                    neighborGem = getGemAt(boardCopy, x + offsetX, y + offsetY)
                    if neighborGem != None and neighborGem in possibleGems:
                        possibleGems.remove(neighborGem)

                newGem = random.choice(possibleGems)
                boardCopy[x][y] = newGem
                dropSlots[x].append(newGem)
    return dropSlots


def findMatchingGems(board):
    gemsToRemove = [] # a list of lists of gems in matching triplets that should be removed
    boardCopy = copy.deepcopy(board)

    # loop through each space, checking for 3 adjacent identical gems
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            # look for horizontal matches
            if getGemAt(boardCopy, x, y) == getGemAt(boardCopy, x + 1, y) == getGemAt(boardCopy, x + 2, y) and getGemAt(boardCopy, x, y) != EMPTY_SPACE:
                targetGem = boardCopy[x][y]
                offset = 0
                removeSet = []
                while getGemAt(boardCopy, x + offset, y) == targetGem:
                    # keep checking if there's more than 3 gems in a row
                    removeSet.append((x + offset, y))
                    boardCopy[x + offset][y] = EMPTY_SPACE
                    offset += 1
                gemsToRemove.append(removeSet)

            # look for vertical matches
            if getGemAt(boardCopy, x, y) == getGemAt(boardCopy, x, y + 1) == getGemAt(boardCopy, x, y + 2) and getGemAt(boardCopy, x, y) != EMPTY_SPACE:
                targetGem = boardCopy[x][y]
                offset = 0
                removeSet = []
                while getGemAt(boardCopy, x, y + offset) == targetGem:
                    # keep checking, in case there's more than 3 gems in a row
                    removeSet.append((x, y + offset))
                    boardCopy[x][y + offset] = EMPTY_SPACE
                    offset += 1
                gemsToRemove.append(removeSet)

    return gemsToRemove



def getDroppingGems(board):
    # Find all the gems that have an empty space below them
    boardCopy = copy.deepcopy(board)
    droppingGems = []
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT - 2, -1, -1):
            if boardCopy[x][y + 1] == EMPTY_SPACE and boardCopy[x][y] != EMPTY_SPACE:
                # This space drops if not empty but the space below it is
                droppingGems.append( {'imageNum': boardCopy[x][y], 'x': x, 'y': y, 'direction': DOWN} )
                boardCopy[x][y] = EMPTY_SPACE
    return droppingGems


def moveGems(board, movingGems):
    # movingGems is a list of dicts with keys x, y, direction, imageNum
    for gem in movingGems:
        if gem['y'] != ROWABOVEBOARD:
            board[gem['x']][gem['y']] = EMPTY_SPACE
            movex = 0
            movey = 0
            if gem['direction'] == LEFT:
                movex = -1
            elif gem['direction'] == RIGHT:
                movex = 1
            elif gem['direction'] == DOWN:
                movey = 1
            elif gem['direction'] == UP:
                movey = -1
            board[gem['x'] + movex][gem['y'] + movey] = gem['imageNum']
        else:
            # gem is located above the board (where new gems come from)
            board[gem['x']][0] = gem['imageNum'] # move to top row


def fillBoard(board):
    dropSlots = getDropSlots(board)
    while dropSlots != [[]] * BOARDWIDTH:
        # do the dropping animation as long as there are more gems to drop
        movingGems = getDroppingGems(board)
        for x in range(len(dropSlots)):
            if len(dropSlots[x]) != 0:
                # cause the lowest gem in each slot to begin moving in the DOWN direction
                movingGems.append({'imageNum': dropSlots[x][0], 'x': x, 'y': ROWABOVEBOARD, 'direction': DOWN})

        boardCopy = getBoardCopyMinusGems(board, movingGems)
        moveGems(board, movingGems)

        # Make the next row of gems from the drop slots
        # the lowest by deleting the previous lowest gems.
        for x in range(len(dropSlots)):
            if len(dropSlots[x]) == 0:
                continue
            board[x][0] = dropSlots[x][0]
            del dropSlots[x][0]


def getBoardCopyMinusGems(board, gems):
    # Creates and returns a copy of the passed board data structure,
    # with the gems in the "gems" list removed from it.
    #
    # Gems is a list of dicts, with keys x, y, direction, imageNum

    boardCopy = copy.deepcopy(board)

    # Remove some of the gems from this board data structure copy.
    for gem in gems:
        if gem['y'] != ROWABOVEBOARD:
            boardCopy[gem['x']][gem['y']] = EMPTY_SPACE
    return boardCopy


if __name__ == '__main__':
    main()
