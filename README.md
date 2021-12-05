# CSE_571_TP
Team Project for CSE 571

*The Pacman training environment is taken from code used in the Berkeley CS188 Course.

Various modifications needed to be made to the following files during development
of our project:
- game.py
- pacman.py
- learningAgents.py

Of note, we removed a memory leak from pacman.py by removing references to the
GameState class variable 'explored'.


Main Project Files:
    Most files contains a "if __name__ == '__main__'" section which contains
    calling code that runs the file's main task.


1) evaluate.py
Contains the routine used to train and evaluate the agents. The bottom portion
of the file contains all of the code necessary to set up the hyper parameters
of the agents and the configuration of the training environment.

The current configuration in the file is setup to train and evaluate all the
agents on the randomized layouts.

2) convergence.py
Contains code used to approximate the episode of congergence for each algorithm
each run. The bottom of the file contains code which dumps out all the values
used in the ANOVA tests.

3) plotting.py
Contains functions which create all the figures used in the report. At the 
bottom of the file there is calling code for each figure which can be
uncommented and run.

4) featureExtractors.py
We implemented our own feature extractor call 'BetterExtractor'

5) layout_gen.py
This file contains two classes. The LayoutGenMultipleBaseLayouts class is used
in evaluate.py in order to randomly modify specified layouts by removing walls,
randomly placing food, ghosts, pacman and capsules.

6) qlearningAgents.py
Contains the implementations of our three RL agent algorithms.

7) generate_plots.py
Another script used for generating plots for comparing batches of runs side
by side.