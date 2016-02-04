##Appendix two details the alternative model based on the code in reactive_agent.py
import pickle, os

from reactive_agent import *

data_dir = "../figures_appendix2/data"
graph_dir = "../figures_appendix2"

##Use this script to produce the graphs for that appendix (and nothing else)
##It will serve as a document to the grpahs, i.e. parameters used etc

##Have it save data so that we can quickly re-run

##3 Model Graphs. Compare the fitness of the three models for a DOL and non-DOL
##Game

def make_3mod_graphs( force = False ):
	"""
	Makes the figures for the comparison of the three models.
	If force is true it will always regenerate the data, other-
	wise it will try to load it from file.

	"""

	print "Making figure for non_DOL game"
	make_3mod_comparison_figure( S = S_non_DOL, T = T_non_DOL,\
	 alphaPoints = 26, reps = 12, force = force, file_format = file_format, long_run = True )

	print "Making figure for DOL game"
	make_3mod_comparison_figure( S = S_DOL, T = T_DOL,\
	  alphaPoints = 26, reps = 12, force = force, file_format = file_format, long_run = True)

##Draw the ST heatmaps for three values of alpha
def make_ST_heatmaps( force = False ):
	"""
	Makes the figures for the comparison of the ST heat maps, these are
	two heatmaps for x_a and beta over all ST space.
	If force is true it will always regenerate the data, other-
	wise it will try to load it from file.

	"""

	for alpha in [0,0.3,0.7,1.0]:
		print "Making heatmap for alpha = {}".format(alpha)
		draw_ST_space( alpha = alpha, points = 41, reps = 12, file_format = 'eps',\
		 long_run = True, force = force, x_points = 11, beta_points = 12)

##Draw the fitness comparison figures for two values of alpha
def make_fitness_maps(force = False):
	"""
	Makes the figures for the comparison of fitness. This is the fitness
	of the full model, minus the fitness of the mixed strategy model.
	If force is true it will always regenerate the data, other-
	wise it will try to load it from file.

	"""

	for alpha in [ 0.3, 1.0 ]:
		pl.figure()
		sweepy.sweep_func( relative_fitness, [ [ 'T', 0, 3, 41 ], [ 'S', -1, 2, 41 ] ], fixed_params = { 'alpha': alpha },\
		 ensure_dir = True, reps = 12, output_directory = os.path.join( graph_dir,"relative_fitness" ,"alpha_{}".format(alpha) ),\
		 look_for_data = not force, file_type = 'eps')


def make_graphs( force = False ):
	""" 
	Make all the graphs. These are 
	
	"""
	#make_3mod_graphs(force)
	make_ST_heatmaps(force)
	#make_fitness_maps(force)

##Set the parameters used for all the plots
S_non_DOL, T_non_DOL = -.5, 1.5
S_DOL, T_DOL = -0.2, 4.0

##In what format to save the figures
file_format = ".eps"

if __name__ == "__main__":
	try:
		flag = sys.argv[1]
		if flag == '-F' or flag == '-f':
			force = True
		else:
			print "Unrecognised flag, will have no effect"
			force = False
	except IndexError:
		force = False
	make_graphs( force )

