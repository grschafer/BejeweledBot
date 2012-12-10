# Greg Schafer
# CSCE 478
# Dec 10, 2012
#
# Bejeweled Bot
#
# Adapted from PyBrain Tutorial:
# http://pybrain.org/docs/tutorial/reinforcement-learning.html

import cPickle
import sys
import argparse

from pybrain.rl.learners import Q


# argparse stuff
# demo or training flag
# number of gems
# animation speed
# starting weights file

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--demo",
                    action="store_true",
                    help="Show the learner playing on a graphical board")
parser.add_argument("-f", "--paramfile",
                    help="Provide a file containing existing training weights \
                    formatted as a Python pickled object (e.g. the \
                    params_trained file in this directory)")
parser.add_argument("-o", "--outfile",
                    help="Save training weights to this file on program exit")
parser.add_argument("-b", "--boardsize",
                    type=int,
                    default=8,
                    help="Set height and width of Bejeweled board, must be at \
                    least 4 (for a 4x4 board); has no effect if --demo flag \
                    not set (all training is done on 4x4); default=8")
parser.add_argument("-g", "--gemtypes",
                    type=int,
                    default=7,
                    choices=xrange(5,8),
                    help="Set number of different color gems, can be between \
                    5-7; default=7")
parser.add_argument("-s", "--speed",
                    type=int,
                    default=25,
                    help="Set animation speed, can be between 1-100; has no \
                    effect if --demo flag not set; default=25")

args = parser.parse_args()

assert 4 <= args.boardsize, "--boardsize parameter must be 4 or larger"
assert 1 <= args.speed <= 100, "--speed parameter must be between 1-100"

if args.demo:
    from gfx.task import BejeweledTask
    from gfx.environment import BejeweledBoard
    from gfx.experiment import Experiment
    from gfx.agent import BejeweledAgent
    from gfx.controller import BejeweledActionValueTable
else:
    from train.task import BejeweledTask
    from train.environment import BejeweledBoard
    from train.experiment import Experiment
    from train.agent import BejeweledAgent
    from train.controller import BejeweledActionValueTable


environment = BejeweledBoard(args.boardsize, args.gemtypes, args.speed)

controller = BejeweledActionValueTable(2**16, 24)
controller.initialize(1.)

if args.paramfile:
    with open(args.paramfile, 'r') as f:
        controller._setParameters(cPickle.load(f))


learner = Q()
agent = BejeweledAgent(controller, learner)


task = BejeweledTask(environment)

experiment = Experiment(task, agent)


try:
    while True:
        experiment.doInteractions(1)
        agent.learn()
        agent.reset()
except:
    if args.outfile:
        with open(args.outfile, 'w') as f:
            cPickle.dump(controller.params, f)
    a, b, c = sys.exc_info()
    raise a, b, c

