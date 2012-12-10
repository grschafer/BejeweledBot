Greg Schafer
CSCE 478
Dec 10, 2012

Project - Reinforcement Learning Bejeweled

========================
===== DEPENDENCIES =====
========================

-PyBrain
-SciPy/NumPy

This code depends on the PyBrain library, which should be installed by running
"pip install PyBrain" or by running the setup.py in the included PyBrain
submodule.

PyBrain code depends on SciPy/NumPy, which I have not included because it
is already installed on cse.unl.edu and is commonly installed with Python
itself.


============================
===== RUNNING THE CODE =====
============================

All run options will print to stdout the following 2 accuracy measurements:

    (boards where a match was made) / (boards with a possible matc)

    (boards where the optimal match was made) / (boards with a possible match)

Running the code with no arguments will begin training a new learner on a
4x4 board.

To specify a starting set of learning weights, use the -f/--paramfile argument
and provide the file containing the Python-pickled weights:

    python rl.py --paramfile params_trained

To save learning weights as a Python-pickled file whenever the program exits,
use the -o/--outfile argument:

    python rl.py --outfile my_saved_params

To graphically show the learner playing Bejeweled, use the -d/--demo flag:

    python rl.py --demo

This --demo flag can be combined with -b/--boardsize and -s/--speed arguments
to control the size of the Bejeweled board that will be played and the speed
of graphical animations, respectively:

    python rl.py --demo --boardsize 16 --speed 75


==============================
===== MORE CMD-LINE HELP =====
==============================

Run the rl.py script with the -h flag to get more help with arguments:

usage: rl.py [-h] [-d] [-f PARAMFILE] [-o OUTFILE] [-b BOARDSIZE] [-g {5,6,7}]
             [-s SPEED]

optional arguments:
  -h, --help            show this help message and exit
  -d, --demo            Show the learner playing on a graphical board
  -f PARAMFILE, --paramfile PARAMFILE
                        Provide a file containing existing training weights
                        formatted as a Python pickled object (e.g. the
                        params_trained file in this directory)
  -o OUTFILE, --outfile OUTFILE
                        Save training weights to this file on program exit
  -b BOARDSIZE, --boardsize BOARDSIZE
                        Set height and width of Bejeweled board, must be at
                        least 4 (for a 4x4 board); has no effect if --demo
                        flag not set (all training is done on 4x4); default=8
  -g {5,6,7}, --gemtypes {5,6,7}
                        Set number of different color gems, can be between
                        5-7; default=7
  -s SPEED, --speed SPEED
                        Set animation speed, can be between 1-100; has no
                        effect if --demo flag not set; default=25

