__author__ = 'Tom Schaul, tom@idsia.ch'

import random
import copy
import numpy as np
from scipy import zeros
from pprint import pformat, pprint

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

# TODO: mazes can have any number of dimensions?

BOARDWIDTH = 4
BOARDHEIGHT = 4
NUMGEMTYPES = 5
assert NUMGEMTYPES >= 5, "numgemtypes > 5, for unique gem drop rule"
GEMTYPES = range(NUMGEMTYPES)
EMPTY_SPACE = -1
ROWABOVEBOARD = 'row above board'
MAX_ITERS = 100

pos = 0
got = 0
opti = 0

# constants for direction values (used for pygame animations)
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class BejeweledBoard(Environment, Named):
    """ 2D mazes, with actions being the direction of movement (N,E,S,W)
    and observations being the presence of walls in those directions.

    It has a finite number of states, a subset of which are potential starting states (default: all except goal states).
    A maze can have absorbing states, which, when reached end the episode (default: there is one, the goal).

    There is a single agent walking around in the maze (Theseus).
    The movement can succeed or not, or be stochastically determined.
    Running against a wall does not get you anywhere.

    Every state can have an an associated reward (default: 1 on goal, 0 elsewhere).
    The observations can be noisy.
    """

    board = None
    score = 0
    gameover = False


    def __init__(self, boardsize, numgemtypes, animspeed, **args):
        global NUMGEMTYPES, GEMTYPES
        assert numgemtypes >= 5, "numgemtypes > 5, for unique gem drop rule"
        NUMGEMTYPES = numgemtypes
        GEMTYPES = range(NUMGEMTYPES)

        self.setArgs(**args)
        self.reset()

    def reset(self):
        """ return to initial position (stochastically): """
        self.board = self._getBlankBoard()
        self._fillBoard(self.board)
        while not self._canMakeMove(self.board):
            self.board = self._getBlankBoard()
            self._fillBoard(self.board)
        self.score = 0
        self.gameover = False

    def _score(self, match, inboard):
        score = 0
        board = copy.deepcopy(inboard)

        firstSelectedGem = {'x': match[0][0], 'y': match[0][1]}
        clickedSpace = {'x': match[1][0], 'y': match[1][1]}

        # Two gems have been clicked on and selected. Swap the gems.
        firstSwappingGem, secondSwappingGem = self._getSwappingGems(board, firstSelectedGem, clickedSpace)

        # Swap the gems in the board data structure.
        board[firstSwappingGem['x']][firstSwappingGem['y']] = secondSwappingGem['imageNum']
        board[secondSwappingGem['x']][secondSwappingGem['y']] = firstSwappingGem['imageNum']

        matchedGems = self._findMatchingGems(board)
        # This was a matching move.
        while matchedGems != []:
            # Remove matched gems, then pull down the board.
            points = []
            for gemSet in matchedGems:
                score += (10 + (len(gemSet) - 3) * 10)
                for gem in gemSet:
                    board[gem[0]][gem[1]] = EMPTY_SPACE

            # Drop the new gems.
            self._fillBoard(board)

            # Check if there are any new matches.
            matchedGems = self._findMatchingGems(board)
        return score

    def _findOptimalMoves(self, board):
        matches = self._possibleMoves(board)
        scores = [self._score(match, board) for match in matches]
        tup = zip(matches, scores)
        maxVal = max(scores)
        maxMoves = filter(lambda x: x[1] == maxVal, tup)
        return [x[0] for x in maxMoves], maxVal


    def performAction(self, action):
        movePos = self._canMakeMove(self.board)

        optiMoves, optiValue = self._findOptimalMoves(self.board)

        scoreAdd = 0
        action = self._actionIndexToSwapTuple(action)
        firstSelectedGem = {'x': action[0][0], 'y': action[0][1]}
        clickedSpace = {'x': action[1][0], 'y': action[1][1]}
        # Two gems have been clicked on and selected. Swap the gems.
        firstSwappingGem, secondSwappingGem = self._getSwappingGems(self.board, firstSelectedGem, clickedSpace)
        if firstSwappingGem == None and secondSwappingGem == None:
            # If both are None, then the gems were not adjacent
            print 'gems not adjacent'
            firstSelectedGem = None # deselect the first gem
            self.lastReward = -10
            return 0

        # Swap the gems in the board data structure.
        self.board[firstSwappingGem['x']][firstSwappingGem['y']] = secondSwappingGem['imageNum']
        self.board[secondSwappingGem['x']][secondSwappingGem['y']] = firstSwappingGem['imageNum']

        # See if this is a matching move.
        matchedGems = self._findMatchingGems(self.board)
        if matchedGems == []:
            #print 'did not cause a match'
            # Was not a matching move; swap the gems back
            self.board[firstSwappingGem['x']][firstSwappingGem['y']] = firstSwappingGem['imageNum']
            self.board[secondSwappingGem['x']][secondSwappingGem['y']] = secondSwappingGem['imageNum']
            self.lastReward = -10
        else:
            # This was a matching move.
            while matchedGems != []:
                # Remove matched gems, then pull down the board.

                for gemSet in matchedGems:
                    scoreAdd += (10 + (len(gemSet) - 3) * 10)
                    for gem in gemSet:
                        self.board[gem[0]][gem[1]] = EMPTY_SPACE
                self.score += scoreAdd

                # Drop the new gems.
                self._fillBoard(self.board)

                # Check if there are any new matches.
                matchedGems = self._findMatchingGems(self.board)
            # TODO: set last reward before combos? otherwise it will get confused
            #  when it gets extra reward
            # combos allowed from pieces already on the board falling into
            # more matches, but not allowed for pieces newly falling into board
            self.lastReward = scoreAdd
        firstSelectedGem = None

        global pos
        global got
        global opti
        if movePos:
            pos += 1
            if scoreAdd > 0:
                got += 1
                if list([action[0], action[1]]) in optiMoves:
                    opti += 1
                print 'found match:', got, '/', pos, '=', \
                      float(got) / pos, 'found optimal:', \
                      opti, '/', pos, '=', float(opti) / pos

        if not self._canMakeMove(self.board):
            #print 'game ended, no more moves available'
            self.gameover = True
            # TODO: tie gameover into episodic learning stuff?
            self.reset()
            return 0



    def getSensors(self):
        indices = self._boardToIndices(self.board)
        return indices

    def getLastReward(self):
        return self.lastReward

    # ====================================================================
    # ==================== BEJEWELED HELPER FUNCTIONS ====================  
    # ====================================================================
    # TODO: add rotation/mirroring support
    def _actionIndexToSwapTuple(self, action):
        """ Converts from action index to tuple of coords of gems to swap """
        # TODO: explain indexing scheme better
        action = int(action[0]) # remove action number from its array
        swapTuple = []
        if action > 11: # vertical swap
            swapTuple.append(divmod(action - 12, 4))
            swapTuple.append((swapTuple[0][0] + 1, swapTuple[0][1]))
        else: # horizontal swap
            swapTuple.append(divmod(action, 3))
            swapTuple.append((swapTuple[0][0], swapTuple[0][1] + 1))
        return tuple(swapTuple)

    def _boardToIndices(self, board):
        """ Converts board to state index for each color (EXPLAIN MORE) 
        Also: ROTATIONS/REFLECTIONS? """
        # TODO: explain indexing scheme better
        b = np.array(board)
        indices = []
        for color in GEMTYPES:
            tmp = np.array(b == color, dtype=int)
            binstr = ''.join((str(i) for i in tmp.flatten()))
            index = int(binstr, base=2)
            indices.append([index]) # TODO: lame that this has to be in a list
        return np.array(indices)

    def _indicesToBoard(self, indices):
        board = np.zeros((4,4))
        for color, index in enumerate(indices):
            s = bin(index[0])[2:]
            s = '0' * (16 - len(s)) + s
            coords = [divmod(i, 4) for i in range(len(s)) if s[i] == '1']
            for c in coords:
                board[c] = color
        return board


    def _getBlankBoard(self):
        # TODO: change to numpy.array
        board = []
        for x in range(BOARDWIDTH):
            board.append([EMPTY_SPACE] * BOARDHEIGHT)
        return board

    def _getSwappingGems(self, board, firstXY, secondXY):
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

    def _canMakeMove(self, board):
        return len(self._possibleMoves(board)) > 0
    def _possibleMoves(self, board):
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
                          ((0,0), (0,2), (0,3),    ((0,0), (0,1))),
                          ((0,0), (0,1), (0,3),    ((0,2), (0,3))))

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
                    if (self._getGemAt(board, x+pat[0][0], y+pat[0][1]) == \
                        self._getGemAt(board, x+pat[1][0], y+pat[1][1]) == \
                        self._getGemAt(board, x+pat[2][0], y+pat[2][1]) != None):
                        moves.append(map(lambda z: (z[0] + x, z[1] + y), pat[3]))
                    if (self._getGemAt(board, x+pat[0][1], y+pat[0][0]) == \
                        self._getGemAt(board, x+pat[1][1], y+pat[1][0]) == \
                        self._getGemAt(board, x+pat[2][1], y+pat[2][0]) != None):
                        moves.append(map(lambda z: (z[1] + x, z[0] + y), pat[3]))
        return moves


    def _pullDownAllGems(self, board):
        # pulls down gems on the board to the bottom to fill in any gaps
        for x in range(BOARDWIDTH):
            gemsInColumn = []
            for y in range(BOARDHEIGHT):
                if board[x][y] != EMPTY_SPACE:
                    gemsInColumn.append(board[x][y])
            board[x] = ([EMPTY_SPACE] * (BOARDHEIGHT - len(gemsInColumn))) + gemsInColumn


    def _getGemAt(self, board, x, y):
        if x < 0 or y < 0 or x >= BOARDWIDTH or y >= BOARDHEIGHT:
            return None
        else:
            return board[x][y]


    def _getDropSlots(self, board):
        # Creates a "drop slot" for each column and fills the slot with a
        # number of gems that that column is lacking. This function assumes
        # that the gems have been gravity dropped already.
        boardCopy = copy.deepcopy(board)
        self._pullDownAllGems(boardCopy)

        dropSlots = []
        for i in range(BOARDWIDTH):
            dropSlots.append([])

        # TODO: remove restriction that there can be no combos from new gems?
        # count the number of empty spaces in each column on the board
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT-1, -1, -1): # start from bottom, going up
                if boardCopy[x][y] == EMPTY_SPACE:
                    possibleGems = list(range(len(GEMTYPES)))
                    for offsetX, offsetY in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        # Narrow down the possible gems we should put in the
                        # blank space so we don't end up putting an two of
                        # the same gems next to each other when they drop.
                        neighborGem = self._getGemAt(boardCopy, x + offsetX, y + offsetY)
                        if neighborGem != None and neighborGem in possibleGems:
                            possibleGems.remove(neighborGem)

                    newGem = random.choice(possibleGems)
                    boardCopy[x][y] = newGem
                    dropSlots[x].append(newGem)
        return dropSlots


    def _findMatchingGems(self, board):
        gemsToRemove = [] # a list of lists of gems in matching triplets that should be removed
        boardCopy = copy.deepcopy(board)

        # loop through each space, checking for 3 adjacent identical gems
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                # TODO: make 3x3 L/T-shape matches work

                # look for horizontal matches
                if self._getGemAt(boardCopy, x, y) == self._getGemAt(boardCopy, x + 1, y) == self._getGemAt(boardCopy, x + 2, y) and self._getGemAt(boardCopy, x, y) != EMPTY_SPACE:
                    targetGem = boardCopy[x][y]
                    offset = 0
                    removeSet = []
                    while self._getGemAt(boardCopy, x + offset, y) == targetGem:
                        # keep checking if there's more than 3 gems in a row
                        removeSet.append((x + offset, y))
                        boardCopy[x + offset][y] = EMPTY_SPACE
                        offset += 1
                    gemsToRemove.append(removeSet)

                # look for vertical matches
                if self._getGemAt(boardCopy, x, y) == self._getGemAt(boardCopy, x, y + 1) == self._getGemAt(boardCopy, x, y + 2) and self._getGemAt(boardCopy, x, y) != EMPTY_SPACE:
                    targetGem = boardCopy[x][y]
                    offset = 0
                    removeSet = []
                    while self._getGemAt(boardCopy, x, y + offset) == targetGem:
                        # keep checking, in case there's more than 3 gems in a row
                        removeSet.append((x, y + offset))
                        boardCopy[x][y + offset] = EMPTY_SPACE
                        offset += 1
                    gemsToRemove.append(removeSet)

        return gemsToRemove



    def _getDroppingGems(self, board):
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


    def _moveGems(self, board, movingGems):
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


    def _fillBoard(self, board):
        dropSlots = self._getDropSlots(board)
        while dropSlots != [[]] * BOARDWIDTH:
            # do the dropping animation as long as there are more gems to drop
            movingGems = self._getDroppingGems(board)
            for x in range(len(dropSlots)):
                if len(dropSlots[x]) != 0:
                    # cause the lowest gem in each slot to begin moving in the DOWN direction
                    movingGems.append({'imageNum': dropSlots[x][0], 'x': x, 'y': ROWABOVEBOARD, 'direction': DOWN})

            boardCopy = self._getBoardCopyMinusGems(board, movingGems)
            self._moveGems(board, movingGems)

            # Make the next row of gems from the drop slots
            # the lowest by deleting the previous lowest gems.
            for x in range(len(dropSlots)):
                if len(dropSlots[x]) == 0:
                    continue
                board[x][0] = dropSlots[x][0]
                del dropSlots[x][0]


    def _getBoardCopyMinusGems(self, board, gems):
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

    def __str__(self):
        """ Ascii representation of the maze, with the current state """
        return pformat(self.board)

