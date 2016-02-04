##This is the version of the model in which is general, except for the fact that genetic 
##assortment is not an evolvable parameter and is instead exogenously contolled.

##Individuals are pair together in a (genetically) assorted manner.
##With probabiity alpha an individual is pair with a genetically identical partner, and with
##probability 1-alpha it is paired with a random individual.

##I want the individuals to be able to specifiy any value of phenotypic assortment, as well as any
#level of strategy frequency. This is achieved by having individuals whoes genotypes are two real
##numbers x and beta. 0<=x<=1 and -1 <= beta <= +1

##Determining phenotype:
##Individuals gain fitness depending on the phentype they have, and upon the phenotype their partner
##has. This is governed by an underlying phenotypic game. One individual is chossen (at random p=1/2)
##to be player one. Player one becomes a B with probability x_i and an A with probability 1-x_i.

##If the individual is selected to be a player 2 it does the following. Firstly, in the case that beta_i
##is positive it becomes the same phenotype as player 1 with probability beta, with probabilty 1-beta
##it uses x_i to determine its phenotype. If beta_i is negative with probabiltiy -beta_i it becomes the
##opposite phenotype to its partner, and with proabability 1+beta it determines its phenotype with x_i.

##Model I and II are special cases of this model. Model II is constrained to have beta_i = 0 for all beta.
##Model I in addition have x_i in {0,1} for all i.

##The model is simulated via the replicator module. We simply need a matix of scores found from the F function
##Replicator module integrates the appropriate equations.

import sys, os

#Add these modules to the python path, or set their locations here.
#sys.path.append("C:\\Users\\sjt4g11\\Desktop\\Dropbox\\Useful Code\\Replicator")
#sys.path.append("C:\\Users\\sjt4g11\\Desktop\\Dropbox\\Useful Code")
import sweepy
import replicator

##This is a file in this directory that defines a nicer colormap to plot with
import my_cmaps

import pickle

import random
import numpy as np
import pylab as pl


import seaborn as sns
import matplotlib as mpl

from parental_ESS import maximal_possible_fitness

class GP_evo:

	"""
	Main class for the evolutionary model in which agents react to the phenotpye of the individual with whom they are paired.
	"""

	def __init__( self, S, T, R = 1, P = 0, x_points = 21, beta_points = 21, alpha = 0, makeGraphs = True, printInfo = True, modNum = 3, long_run = False ):

		"""
		Params
		======

		S : float
			The vaule of S in the RSTP payoff matrix

		T : float
			The value of T in the RSTP payoff matirx

		R : float, optional {1}
			R in the RSTP payof matix

		P : float, optional {0}
			P in the RSTP payoff matrix

		x_points : int, optional {21}
			The number of descrete points to break the genetic value of x interpolation

		beta_points : int, optional {21}
			Number of descrete points to break the genetic value of beta into

		alpha : float, optional {0}
			The level of genetic assortment, this is really Hamilton's r.

		makeGraphs : bool {True}
			Whether or not to output the makeGraphs

		printInfo : bool {True}
			Print information on the progress and final result

		modNum : int {3,2,1}
			Specify the integer of the model number, 3 is the full model, 2 the mixed strategy model, and
			1 the pure strategy model. See paper for more information.

		long_run : bool {False}
			Multiply the number of generations integrated over by 10 if this is false. This is to really
			ensure convergence, it is usually overkill but possibly recomended for final figures destined
			for publication. Note this is simply passed to the replicator module.

		See Also
		========

		`parental_agent.py` and `parental_ESS.py`. Modules (and classes) for a similar version of the model where
		individuals' phenotpye is determined by the phenotype of their parents. 

		"""

		self.S = S
		self.T = T
		self.R = R
		self.P = P
		self.M = np.array( [ [R,S], [T,P] ] )
		self.x_points = x_points
		self.beta_points = beta_points
		self.alpha = alpha
		self.makeGraphs = makeGraphs
		self.printInfo = printInfo
		##Two special cases of the model (see paper)
		self.modNum = modNum
		self.long_run = long_run

	def F_1_pos( self, I1, I2 ):
		"""
		The expected score of individual one against individual two given that individual one is assigned to the role of player 1.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""
		
		i,j = I1
		k,l = I2

		return ( 1 - l )*( ( 1 - i )*( ( 1 - k )*self.R + k * self.S ) + i*( ( 1 - k )*self.T + k*self.P ) )\
			+ l*( ( 1 - i )*self.R + i*self.P )

	def F_1_neg( self, I1,I2):
		"""
		Expected score that a player receives on being player one, and having a positive value of beta.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""

		i,j = I1
		k,l = I2
		
		return ( 1 + l )*( ( 1 - i )*( ( 1 - k )*self.R + k * self.S ) + i*( ( 1 - k )*self.T + k*self.P ) )\
			- l*( ( 1 - i )*self.S + i*self.T )

	def F_1(self,I1,I2):
		"""
		Expected score on being player one. This is just F_1_neg and F_1_pos rapted in an if statement.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""

		if I2[1] > 0:
			return self.F_1_pos( I1, I2 )
		else:
			return self.F_1_neg( I1, I2 )

	def F_2_neg( self, I1, I2 ):
		"""
		Expected payoff on being player two and having a negative beta.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""

		i,j = I1
		k,l = I2

		return -j*( ( 1 - k )*self.T + k*self.S )\
			+ ( 1 + j )*( ( 1 - k )*( ( 1 - i )*self.R + i*self.T ) + k*( ( 1 - i )*self.S + i*self.P ) )

	def F_2_pos( self, I1, I2 ):
		"""
		Expected payoff on being player 2 and having a positive beta.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""

		i,j = I1
		k,l = I2

		return j*( ( 1 - k )*self.R + k*self.P )\
			+ ( 1 - j )*( ( 1 - k )*( ( 1 - i )*self.R + i*self.T ) + k*( ( 1 - i )*self.S + i*self.P ) )

	def F_2(self,I1,I2):
		"""
		Expcted payoff on being player 2.
		Given the individual 1 has genotype I1 = x1, beta 1, and individual 2 has genotype I2 = x2, beta2.
		"""

		if I1[1] > 0:
			return self.F_2_pos( I1, I2 )
		else:
			return self.F_2_neg( I1, I2 )

	def F( self, I1, I2 ):
		"""
		Expected payoff, simply 1/2*F_1 + 1/2*F_2

		"""

		return .5*self.F_1( I1, I2 ) + .5*self.F_2( I1, I2 )

	def graph_data(self):
		"""
		Makes a pair of figures for the outcome of the model. Both of these are heatmaps. The first 
		being the density of strategies in x,beta space, and the second being a heatmap of average phenotypic
		assortment and average value of the "A" phenotpye.

		If modnum == 1 or 2 then this is simply a line graph showing x_A over time. 

		"""

		pl.figure()
		if self.modNum == 1:
			pl.plot( self.rep.y )
			pl.xlabel( 't' )
			pl.ylabel( "Frequency" )
			pl.legend( [ '$x_A$', '$x_B$' ] )
		elif self.modNum == 2:
			pl.plot( self.xs, self.data[0] )
			pl.xlabel( 'x' )
			pl.ylabel( "Frequency" )
		elif self.modNum == 3:
			pl.figure()
			pl.imshow( self.data, interpolation = 'nearest', extent = [self.xs[0], self.xs[-1],self.betas[0],self.betas[-1]  ], 
				origin = [ self.xs[0], self.betas[0] ], aspect = 'auto', vmin = 0, vmax = 1 )
			pl.xlabel('$x_i$',  fontsize = 40)
			pl.ylabel('$\\beta_i$',  fontsize = 40)
			pl.colorbar()

			##Make the "movement through triangle space" figure
			pl.figure()
			pl.plot( [0,.5],[0,1], color = 'black', linewidth = 5 )
			pl.plot( [0.5,1],[1,0], color = 'black', linewidth = 5 )
			xa_s = np.linspace(0,1,101)
			pl.plot( xa_s, map(self.phi_R,xa_s), '--', color = 'black', linewidth = 2.5 )
			pl.xlabel( '$x_A$', fontsize = 30 )
			pl.ylabel( '$\\varphi$', fontsize = 30 )
			pl.plot( self.phase_data[:,0], self.phase_data[:,1], 'o-' )
			pl.ylim([ 0, 1.05 ])
			pl.xlim( [ 0, 1 ] )

			# ess = self.ESS()
			# pl.annotate('ESS', fontsize = 30, xy=( ess , self.phi_R(ess) ), xytext=(.1,.8),
   #          arrowprops=dict(facecolor='black', shrink=0.05, linewidth = 0.001),
   #          )#

			# ses = self.SOF()
			# pl.annotate('SOF', fontsize = 30, xy=( ses , self.phi_R( ses )), xytext=(.9,.5),
			#             arrowprops=dict(facecolor='black', shrink=0.05, linewidth = 0.001),
			#             )

	def phi_R(self, x):
		"""
		The expected value of phi given that interactions are random. This is useful for comparison graphs.
		Phi is the frequency of interactions that are between unlike types. See paper for more info.

		"""

		return 2*x*(1-x)

	def means(self):
		"""
		Returns the mean of x and beta of the population.

		"""
		mean_b = np.sum(self.data,1).dot( self.betas )
		mean_x = 1 - np.sum(self.data,0).dot( self.xs )
		return mean_x, mean_b

	def phase_space(self):
		"""
		Returns an array of phase space states over time in the form ( x_a(t), phi(t) )
		i.e. a parametric output of the model. This is not used in the central code, but is
		useful for experimentation.

		"""
		self.phase_data = np.zeros( ( len(self.rep.y) , 2 ) )
		for i,state in enumerate( self.rep.y ):
			state.resize( (self.beta_points,self.x_points) )
			x_bar = np.sum( state, 0 ).dot( self.xs )
			##the frequency of individuals with a negative beta
			f_beta_neg = np.sum( state[ :self.beta_points/2 , : ] )
			f_beta_pos = 1 - f_beta_neg
			##Mean negaitive beta
			mean_beta_neg = np.sum( state[ :self.beta_points/2 , : ],1 ).dot( self.betas[:self.beta_points/2] )
			##Mean non-negative beta
			mean_beta_pos = np.sum( state[ self.beta_points/2: , : ],1 ).dot( self.betas[self.beta_points/2:] )
			##Frequency of unlike phenotypic interactions
			phi = 2*( 1 - mean_beta_pos*f_beta_pos )*( 1 - x_bar )*x_bar +\
				mean_beta_neg*f_beta_neg*( 2*( 1 - x_bar )*x_bar - 1 )
			#Frequency of phenotype x
			xa = 1 - x_bar - mean_beta_neg*f_beta_neg*( x_bar - 0.5 )
			self.phase_data[i,:] = [ xa, phi ]

 
	def go(self):
		"""
		Sets the model in motion by phrasing it in terms of a evolutionary game and calling the replicator module
		on it.

		"""

		if self.modNum == 1:
			self.betas = [0]
			self.beta_points = 1
			self.xs = [0,1]
			self.x_points = 2
		elif self.modNum == 2:
			self.betas = [0]
			self.beta_points = 1
			self.xs = np.linspace( 0, 1, self.x_points )
		elif self.modNum == 3:
			self.betas = np.linspace( -1, 1, self.beta_points )
			self.xs = np.linspace( 0, 1, self.x_points )

		##List of genotypes of all strategies being considered
		self.all_strats = []
		for beta in self.betas:
			for x in self.xs:
				self.all_strats.append( (x,beta) )

		##Payoff matrix
		self.M_all = np.array( [ [ self.F(I1,I2) for I2 in self.all_strats ] for I1 in self.all_strats ] )

		self.rep = replicator.Replicator( self.M_all, alpha = self.alpha, y0 = 'random', makeGraphs = False, printInfo = False, long_run = self.long_run )
		self.data = self.rep.finalState
		self.data.resize( (self.beta_points,self.x_points) )

		self.phase_space()

		self.mean_x,self.mean_b = self.means()
		self.finalFit = self.rep.finalFit

		tol = 1e-3
		if np.sum(self.data) > 1 + tol or np.sum(self.data) < 1 - tol:
			print "Warning, frequencies sum to %.3f"%np.sum(self.data)

		if self.printInfo:
			print "Mean x = ",self.mean_x
			print "Mean beta = ",self.mean_b
			print "Final Fitness = ", self.finalFit
			
		if self.makeGraphs:
			self.graph_data()
			pl.show()

data_dir = "../figures_appendix2/data"
graph_dir = "../figures_appendix2"
def make_3mod_comparison_figure(S,T, alphaPoints = 11, reps = 6, force = False, file_format = 'png', output = True, **kwargs):
	"""
	Sweeps through a alpha (aka r) for a given game, specified by S and T. It does this for all three versions of the model.
	1 = pure strategy, 2 = mixed_strategy and 3 =full model.
	The outcome is a line graph plotting final mean fitness for each of the models (three lines).
	For the third model it also plots the average value of beta and x.
	
	Params:
	=======
	
	S and T : floats
		Specify the game being played.
	alphaPoints : int {11}
		Number of point to plot alpha at.
	reps : int {6}
		Number of times to repeat and take the mean over.
	force : bool {False}
		This function will look for saved data if it exists rather than regenerating
		from scratch. Set this to True to force the data to be reloaded even if it
		exists.
	file_format str {'png'} :
		Format in which to save the file, will be passed to pyplot.savefig
	output : bool {True}
		set to False to stop the saving of data and figures
	kwargs :
		Additional keyword arguments passed to GP_evo.
		Note that some are set by default and so specifying them
		here will result in an error. They are: alpha, printInfo,
		makeGraphs and modNum.

	Returns
	=======

	Null


	"""

	file_suffix = "3_mods"
	data_file = os.path.join( data_dir, "{}_S_{}_T_{}.p".format( file_suffix, S, T ) )
	data_exists = os.path.exists( data_file )

	alphas = np.linspace(0,1,alphaPoints)

	##First look at the non-DOL version
	##If force is False and data exists load the data
	if not force and data_exists:
		print "Loading data from file " + data_file
		##Load data
		data = pickle.load( open( data_file, 'rb' ) )

	##Otherwise generate it from scratch
	else:

		print "Gennerating data"
		mean_xs = np.zeros( (alphaPoints, reps) )
		mean_betas = np.zeros( (alphaPoints, reps) )

		alphas = np.linspace(0,1,alphaPoints)
		fitness_data = np.zeros( (3,alphaPoints,reps) )
		for k in xrange(reps):
			print "Rep:",k
			for j,alpha in enumerate(alphas):

				# print alpha
				for modNum in [1,2,3]:

					##This is a bit of a fudge in order to get the model to repeat if the integration blows up.
					valid = False
					tol = 1e-3
					while not valid:

						model = GP_evo( S, T, alpha = alpha, printInfo = False, makeGraphs = False, modNum = modNum, **kwargs)
						model.go()
						fitness_data[ modNum - 1, j, k] = model.finalFit

						if modNum == 3:
							mean_xs[j,k] = model.mean_x
							mean_betas[j,k] = model.mean_b

						##Check to see whether the model has outputted a valid disribution
						##(i.e. frequencies sum to one).
						if np.sum(model.data) > 1 + tol or np.sum(model.data) < 1 - tol:
							valid = False
						else:
							valid = True

		#put all this data in a tuple so it's eassier to pickle
		data = ( fitness_data, mean_xs, mean_betas )
		if output:
			##Write the data to a pickle file
			print "Saving data to file " + data_file
			pickle.dump( data, open( data_file, 'wb') )
						
	##Unpack the array
	fitness_data, mean_xs, mean_betas = data
	pl.figure()
	mean_fitness_data = np.mean(fitness_data,2)
	pl.subplot(212)
	for i in range(3):
		pl.plot(alphas, mean_fitness_data[i,:], 'o-')
	#pl.ylim([-1,1])
	pl.xlabel( '$r$', fontsize = 30 )
	pl.ylabel( '$\\bar{\\pi}$', fontsize = 30 )
	pl.legend( ['Pure', 'Mixed', 'Reactive'], loc = 4)
	pl.plot( [0,1], [1,1], '--', color = 'blue' )
	if S + T > 2:
		pl.plot( [0,1], [pi_SES(S,T)]*2, '--', color = 'green')
		pl.plot( [0,1], [(S+T)/2.]*2, '--', color = 'red')
		pl.ylim([-.1,( (S+T)/2. )*1.1])
	else:
		pl.ylim([-.1,1.1])

	pl.subplot(211)
	pl.plot( alphas, np.mean(mean_xs,1) , 'o-', label = '$x_A$' )
	pl.plot( alphas, np.mean(mean_betas,1), 'o-', label = '$\\bar{\\beta}$')
	#pl.xlabel( '$\\alpha$', fontsize = 30 )
	pl.ylabel( '$x_A$, $\\bar{\\beta}$', fontsize = 30 )
	pl.ylim([-1.1,1.1])
	if S+T>2:
		pl.legend(loc = 1)
	else:
		pl.legend(loc = 4)

	figure_file = os.path.join( graph_dir, "{}_S_{}_T_{}.{}".format( file_suffix, S, T, file_format ) )
	if output:
		print "Saving figure to file " + figure_file
		pl.savefig( figure_file )
##This is for backward compatability
make_figure = make_3mod_comparison_figure

def pi_SES(S,T):
	"""
	Returns the fitness obtained at the (unstructured) SES.
	Useful for comparisons with models.

	"""
	return ( S + T )**2/( 4*( S + T - 1 ))

def draw_ST_space(alpha = 0, debug = False, points = 11, reps = 2, force = False, file_format = 'png', output = True, **kwargs):
	"""
	Takes model III and plots the results over all of ST space. Two heat maps
	are created, one for x_bar and one for beta_bar.

	Params
	======
	alpha : float [0,1] {0} :
		alpha, i.e. r genetic assortment.
	debug : bool {False}
		Set to true to very quickly run the model to see if it is working.
	points : int {11}
		Number of descrete points to break up the space into.
	reps : int {2}
		Number of times to repeat the model.
	force : bool {False}
		The function will pickle the outputs, set to true to force the
		function to regenerate the data.
	file_format : str {'png'}
		Type of figure file to save. Anything that matplotlib will accept
	output : bool {True}
		Set to False to not save data and figure. You'll only be able to
		see the outputs if you're running interactively.
	kwargs :
		Additional keyword arguments passed to GP_evo model.

	"""

	label_size = 18
	tick_label_size = 15
	legend_label_size = 13

	##We don't want this to be the integer 1 or 0, it messes with the formatting!
	alpha = float(alpha)
	
	Ss = np.linspace(-1,4,points)
	Ts = np.linspace(0,4,points)

	beta_data_file = os.path.join( data_dir, "ST_sweep_beta_%.2f"%alpha )
	x_data_file = os.path.join( data_dir, "ST_sweep_x_%.2f"%alpha )

	#See if the data has been generated already
	try:

		##Bit of a hack
		if force:
			raise IOError

		f1 = open( beta_data_file,'rb')
		data_beta_bar = pickle.load( f1 )
		f1.close()


		f2 = open( x_data_file, 'rb' )
		data_x_bar = pickle.load( f2)
		f1.close()

		print "Data Found"

	except IOError:

		print "No Data Found, generating ..."

		data_x_bar = np.zeros( ( points, points, reps ) )
		data_beta_bar = np.zeros( ( points, points, reps ) )
		for k in xrange(reps):
			print "Rep:",k + 1
			for i,S in enumerate( Ss ):
				for j,T in enumerate( Ts ):
					#print S,T
					if debug:
						mod = GP_evo(S,T, alpha = alpha, x_points = 3, beta_points = 2, makeGraphs = False, printInfo = False, **kwargs)
					else:
						mod = GP_evo(S,T, alpha = alpha, makeGraphs = False, printInfo = False, **kwargs)
					mod.go()
					data_x_bar[i,j,k] = mod.mean_x
					data_beta_bar[i,j,k] = mod.mean_b

		##Save the data
		if output:
			print "Saving data at " + beta_data_file
			f1 = open( beta_data_file,'wb')
			pickle.dump( data_beta_bar, f1 )
			f1.close()

			print "Saving data at " + x_data_file
			f2 = open( x_data_file,'wb')
			pickle.dump( data_x_bar, f2)
			f2.close()

	##Create a red green blue array for x data 
	# x_mean = np.mean( data_x_bar, 2 )
	# RGB_data = np.array( [[ [ 1 - x_mean[i,j], 0 , x_mean[i,j], 0.1 ]  for i in xrange(points) ] for j in xrange(points) ] )

	#cmap = pl.get_cmap('Reds')

	##Create a colour map using seaborn
	cmap_linear = mpl.colors.ListedColormap( sns.light_palette((210, 90, 60), input="husl", n_colors = 101) )

	##For diverging colour
	cmap_diverging = mpl.colors.ListedColormap( sns.diverging_palette(10, 220, sep=80, n=101)  )

	f = pl.figure()
	pl.subplot(121)
	pl.imshow( np.mean( data_x_bar, 2), origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
		extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0, vmax = 1, cmap = cmap_linear )
	# pl.imshow( RGB_data, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
	#  	extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0, vmax = 1, cmap = cmap)
	pl.title('$\\bar{x}$', fontsize = label_size)
	pl.xlabel('T', fontsize = label_size)
	pl.ylabel('S', fontsize = label_size)
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	cb = pl.colorbar(shrink = .62, ticks = np.linspace( 0, 1, 6 ) )
	cb.ax.tick_params(labelsize=legend_label_size)
	##Fidle with the axis lables
	pl.tick_params(axis='both', which='major', labelsize=tick_label_size)
	pl.locator_params(axis = 'x', nbins=4)
	##Turn grid lines off
	pl.grid('off')

	pl.subplot(122)
	pl.imshow( np.mean( data_beta_bar, 2), origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
	 extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = -1, vmax = 1, cmap = cmap_diverging )
	pl.title('$\\bar{\\beta}$', fontsize = label_size)
	pl.xlabel('T', fontsize = label_size)
	#pl.ylabel('S', fontsize = label_size)
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	cb = pl.colorbar(shrink  = .62, ticks = np.linspace( -1, 1, 6 ))
	cb.ax.tick_params(labelsize=legend_label_size)
	##Fidle with the axis lables
	pl.tick_params(axis='both', which='major', labelsize=tick_label_size)
	pl.locator_params(axis = 'x', nbins=4)
	##Turn grid lines off
	pl.grid('off')

	fig_file = os.path.join( graph_dir, "alpha_{:.2}.{}".format( alpha, file_format) )
	if output:
		print "saving figure at " + fig_file
		pl.savefig( fig_file, bbox_inches='tight')
	else:
		return f

def examples(AP = 21, reps = 12, output = False):
	"""
	Runs make figure on a number of standard games. Altogether will make
	one for each of the 4 fundamental games as well as for DOL -non-DOL variants.
	This makes 7 in total as there is no-stag-hunt DOL game.

	output : bool {False}
		Will save the data and figure if True

	"""

	print "A non-DOL Harmony Game"
	S,T = 0.4,0.7
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

	print "A DOL harmony Game"
	S,T = 1.8, 0.4
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

	print "A Staghunt game"
	S,T = -.4, 0.7
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

	print "A non-DOL Snowdrift game"
	S,T = 0.4, 1.4
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

	print "A DOL snowdrift Game"
	S,T = -0.6, 1.7
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output )

	print "A non-DOL PD"
	S,T = -0.2,1.6
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

	print "A DOL PD"
	S,T = -.2, 4.
	make_figure(S,T, alphaPoints = AP, reps = reps, output = output)

def relative_fitness(S,T, alpha = 0):
	"""
	Returns the relative fitness of the full model, compared with the
	mixed strategy model, for a given value of S and T.

	Params
	======
	S and T : floats
		Specify the game
	alpha : float [0,1] {0}
		Relatedness, aka genetic assortment aka r.

	"""
	
	##Run the full model and record the fitness

	##The integration seems to blow up periodically, let's hack this by trying repeatedly until we get a sensible value!
	relative_fitness = -1000
	count = 0
	while relative_fitness < -10 or relative_fitness > 10: 
		model = GP_evo( S, T, alpha = alpha, x_points = 12, beta_points = 11, printInfo = False, makeGraphs = False, modNum = 3)
		model.go()
		full_fitness = model.finalFit

		##Run the mixed strategy model and record the fitnss
		model = GP_evo( S, T, alpha = alpha, x_points = 12, beta_points = 11, printInfo = False, makeGraphs = False, modNum = 2)
		model.go()
		mixed_fitness = model.finalFit

		relative_fitness = full_fitness - mixed_fitness

		count += 1
		if count > 1:
			print "It looks like the integration has blown up, trying again!"
			print count
		if count > 12:
			print "Could not find sensible value for relative fitness, setting equal to zero and breaking!"
			relative_fitness = 0
			break

	##Return the difference in these two things
	return relative_fitness

def ST_pi_sweep(r, steps = 5):
	"""
	Sweeps through S and T and plots the fitness divided by the maximum possible fitness.
	
	inputs
	======
	r : float
		Relatedness
	steps : int
		Number of decrete points to split S and T into

	"""

	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			print S,T
			mod = GP_evo(S,T, alpha = r, makeGraphs = False, printInfo = False)
			mod.go()
			dataFull[i,j] = mod.finalFit/maximal_possible_fitness( S, T )

	#pl.figure()
	cmap = pl.get_cmap('Reds')
	pl.imshow(dataFull, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
			extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], cmap = cmap, vmin = 0, vmax = .5*(Ss[-1]+Ts[-1]))
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	#pl.colorbar()
	#pl.show()

	return dataFull

##Plots an alternative figure for the version in which I plot the fitness achieved for the
##full model relatve to the maximum possible, for both the parantal and the mixed strategy
##version.
if __name__ == "__main__":

	#Directory in which to pickle data
	data_dir = "../additional_figures/data"
	fig_dir = "../additional_figures"
	from parental_ESS import ST_pi as ST_pi_parental
	##Let's plot relatvie fitness for a few values of r,
	#and comapre to the other model
	pl.figure()
	steps = 5
	ST_points = 21
	for i,r in enumerate( np.linspace(0,1,steps) ):
		
		pl.subplot( 2, steps, i + 1 )
		data = ST_pi_sweep(r, ST_points)
		##Pickle this data to file
		file_name = os.path.join( data_dir, "r_{}_points_{}.p".format( r, ST_points ) )
		pickle.dump( data, open( file_name, 'wb') )

		pl.subplot( 2, steps, steps + i + 1 )
		ST_pi_parental(r)

	pl.savefig( os.path.join( fig_dir, "reactive_v_parental_steps_{}.eps".format( steps ) ) )