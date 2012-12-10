from pybrain.rl.learners.valuebased.interface import ActionValueTable
from scipy import where
from random import choice
import numpy as np


# jank jank jank
# TODO: maybe find a better way to do this?
class double(np.float64):
    def __new__(cls, arg=0.0):
        return np.float64.__new__(cls, arg)
    def __init__(self, arg=0.0):
        self.value = arg

class BejeweledActionValueTable(ActionValueTable):

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        values = self.params.reshape(self.numRows, self.numColumns)[state, :].flatten()
        maxVal = max(values)
        action = where(values == maxVal)[0]
        action = choice(action)
        self.lastMaxActionValue = maxVal
        return action

    def printState(self, state):
        print self.params.reshape(self.numRows, self.numColumns)[state, :].flatten()
