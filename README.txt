All the figures from the paper, including the two appendices can be created from the code and from the data contained in this repo. Python2 is required. All code is in the folder code, data can either be generated from scratch from each script, or can be loaded from the files. Most of the code is intelligent enough to look for data before trying to generate it from scratch.

Data is in the pickle format, and needs python to unpickle it.

In addition the mathematica notebook "SI_ESS_calculator" deals with finding the ESSs of the main model.

The code contains the files

makeTraingles.py :
	Simple static code for creating the phase space diagram in the introduction

parental_ESS.py :
	Defines the equations of the ESSs of the main model for the paper.

parental_agent.py :
	Defines the classl used to simulate the main model via an semi-agent based approach

reactive_agent.py :
	This is the alternative formulation of the model which is described in the appendix

make_graphs_for_appendix2.py
	Calls the necessary modules to make the graphs in the appendix

In order to be able to run the code you need to install some python modules. Do this with pip install -r requirments.txt

You will also need the two modules sweepy and replicator, which are both available on my github page:
see here:

	https://github.com/simontudge

Clone these two repositories and add their location to the python path.

Sweepy is a tool for automatic parameter sweeps of models, and replicator is a tool for studying evolutionary game theory.