__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.environments import Task
from scipy import array

class BejeweledTask(Task):
    """ This is a MDP task for the MazeEnvironment. The state is fully observable,
        giving the agent the current position of perseus. Reward is given on reaching
        the goal, otherwise no reward. """

    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        # reward = score assigned by environment
        return self.env.getLastReward()

    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given
            to the super class.
        """
        # action = tuple of elements to swap
        Task.performAction(self, action)


    def getObservation(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        # obs = board matrix
        return self.env.getSensors()



