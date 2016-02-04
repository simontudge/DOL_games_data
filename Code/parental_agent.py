"""Version of the model in which individuals can control the phenotype of their offspring.
This is done with a certain probability r. One can think of this as the probability of
interacting with ones offspring, and having a certain reproductive strategy.

An individual's payoff can be broken down into a the payoff recieved on meeting an individual pi_self. And
the payoff recieved upon meeting a random individual, pi_random. Thus, pi = r*pi_self + (1-r)*pi_random.

Individuals have a genotype of two real numbers so individual i's genotype G_i is given by [xi_1,xi_2].
x_1 is the probability of playing strategy B and x_2 is the proability of creating an offspring with phenotype B.

The function F[p,q] is the expected payoff of a mixed strategy individual with strategy p, playing an individual
with strategy q.

We assume that when two individuals interact they have equal proability of being the one who determiens the phenotype of
the other player. This leads to:

		pi_self[i,j] = 1/2*F[i,j] + 1/2*F[j,i]

And: 	pi_random = F[i,\bar{x_1}], where \bar{x_1} is the population mean of x_1.

See also
========
parental_ESS.py :
	Python file for plotting the ESS of the game. For this model I have worked out the ESSs mathematically so it is much quicker
	to use the other version.
reactive_agent.py
	Python file for the alternative formulation of the model in which individuals choose a phenotype based on the phenotype of 
	the individual with whom they interact.

"""

import random
import numpy as np
import pylab as pl
import pickle
import os

import scipy.integrate as I


class GP_evo3:
	"""
	This is a class representing the main model, in which we evolve individuals who can determine the phenotype
	of the their offspring. Call the go method in order to begin integration.

	Example Use:
	============

		>>> my_model = GP_evo3( 0.3, 1.4, r = 0.3, makeGraphs = True )
		>>> my_model.go()

	Displays graphs and info to stdout.

	"""
	
	def __init__( self, S, T, R = 1 ,P = 0, r = 0, steps = 11, t_f = 201, t_steps = 1001, makeGraphs = True,\
	 printInfo = True, modelVersion = 3, saveFigs = False):
		r"""
		Init of GP_evo3

		Basic setup of the class, call go method to set running after Initialising.

		Inputs
		======

		S : float
			Parameter of the game
		T: float
			Parameter of the game
		R : float {1}
			Parameter of the game
		P : float {0}
			Parameter of the game
		r : float [0,1] {0}
			Value of the relatedness i.e. genetic assortment
		steps : int {11}
			The number of decrete steps to break up the genotype space into
		t_f : float {201}
			The time at which the integration will run to
		t_steps : int {1001}
			The number of desrete time points at which to plot (and store) the outcome of integration
		makeGraphs : bool {True}
			Whether or not to output graphs
		printInfo : bool {True}
			Print some basic info at the end of the integration if set to True
		model_version int {3}:
			May be 1, 2 or 3. Which version of the model to run. 3 is the full model, 2 is a special case
			restricted to only mixed strategies with \beta = 0. 1 is the special case with only pure strategies
		saveFigs : bool {False}
			Figures will be saved to the standard location if set to true. The default location is "Figures/Example Runs V3"

		"""

		if r < 0. or r > 1.0:
			raise ValueError ("r must lie between 0 and 1, got {}".format( r ) )

		self.R,self.S,self.T,self.P = R,S,T,P
		self.r = r
		if modelVersion == 1:
			self.steps = 2
		else:
			self.steps = steps
		self.t_steps = t_steps
		self.t_f = t_f
		self.makeGraphs = makeGraphs
		self.printInfo = printInfo
		self.modelVersion = modelVersion
		self.saveFigs = saveFigs

		self.xs = np.linspace( 0, 1, self.steps )

		##an array of x1*x2 for use in calculating covariances
		self.x1_x2 = [ [ x1*x2 for x1 in self.xs ] for x2 in self.xs ]
		
		##The directory for saving figures
		if self.saveFigs:
			self.fig_folder = "Figures\\Example Runs V3\\%.2f_%.2f\\%.2f"%(self.S,self.T,self.r)
			if not os.path.exists(self.fig_folder):
				os.makedirs( self.fig_folder )


	def F(self,p,q):
		"""Expected payoff individual playing mixed strategy p receives on meeting individual playing with mixed strategy q"""
		return p*q*self.P + p*(1-q)*self.T + (1-p)*q*self.S + (1-p)*(1-q)*self.R

	def payoff_self(self,x1,x2):
		"""Payoff to an individual playing with strategy x1,x2. Assuming that the player meets itself"""

		return 0.5*( self.F( x1, x2 ) + self.F( x2, x1 ) )

		#return self.F( x1, x2 )

	def payoff(self,x1,x2,mx):
		"""Total payoff a player receives"""
		return self.r*self.payoff_self(x1,x2) + (1-self.r)*self.F(x1,mx)

	def df(self,f,t):
		"""Differential change in the frequency"""
		f.resize( (self.steps, self.steps) )
		mx = np.sum(f,1).dot(self.xs)
		payoffs = np.array( [ [ self.payoff(x2,x1,mx) for x1 in self.xs] for x2 in self.xs ] )
		mean_payoff = np.sum( payoffs*f )
		delta_f = f*( payoffs - mean_payoff )
		delta_f.resize( self.steps**2 )
		return delta_f

	def PI_bar(self,f):
		"""
		Returns the avergae fitness.

		Note
		=====
		This is calculated at every time step, however, this function is used to extract this value from the final data

		"""

		f.resize( (self.steps, self.steps) )
		mx = np.sum(f,1).dot(self.xs)
		payoffs = np.array( [ [ self.payoff(x2,x1,mx) for x1 in self.xs] for x2 in self.xs ] )
		return np.sum( payoffs*f )

	def return_data( self, f ):
		"""Given f in data form this returns mean_x1, mean_x1 and beta"""

		f.resize( ( self.steps, self.steps ) )

		Ex1 = np.sum(f,1).dot(self.xs)
		Ex2 = np.sum(f,0).dot(self.xs)

		Ex1_sq = np.sum( f, 1 ).dot(self.xs**2)

		E_x1_x2 = np.sum( f*self.x1_x2 )

		##The expected frequency from random interactions:
		phi_random = self.r*( Ex1 + Ex2 - 2*Ex1*Ex2 ) + 2*( 1 - self.r )*( Ex1 - Ex1**2 )

		##The frequency of unlike interactions
		phi = self.r*( Ex1 + Ex2 - 2*E_x1_x2 ) + 2*( 1 - self.r )*( Ex1 - Ex1**2 )

		##Phenotypic linkage disequilibrium
		LD = 1 - phi/phi_random

		xA = 1 - 0.5*self.r*( Ex1 + Ex2 ) - ( 1 - self.r )*Ex1
		
		return Ex1,Ex2,phi,xA

	def phi_R(self, x):
		"""
		The value of \phi for given random interactions, see paper for more detail.
		"""

		return 2*x*(1-x)

	def go(self):
		"""
		Set the integration in motion.

		"""

		##We model the situation by splitting genotype space into dicrete chunks.
		#Freqs is the frequency of each genotype.
		#f0 = np.array( [ [ 1/float(steps**2) for _ in xrange(steps)] for __ in xrange(steps) ] )
		if self.modelVersion == 3:
			self.f0 = np.array( [ [ random.random() for _ in xrange(self.steps)] for __ in xrange(self.steps) ] )
			self.f0 = self.f0/np.sum(self.f0)
		elif self.modelVersion == 2:
			self.f0 = np.zeros( ( self.steps, self.steps ) )
			for i in xrange(self.steps):
				self.f0[i,i] = random.random()
			self.f0 = self.f0/np.sum(self.f0)
		elif self.modelVersion == 1:
			rand = random.random()
			self.f0 = np.array( [ [ rand , 0 ],[0, 1 - rand ]  ] )
		self.f0.resize( self.steps**2 )
		
		self.ts = np.linspace(0,self.t_f,self.t_steps)

		##Perform the integration
		self.f_of_t = I.odeint( self.df, self.f0, self.ts )

		##Calculate mean x1, mean x2 and mean beta from data
		self.all_data = np.array( map(self.return_data, self.f_of_t) )
		self.x1_means = self.all_data[:,0]
		self.x2_means = self.all_data[:,1]
		# self.betas = self.all_data[:,2]
		self.phis = self.all_data[:,2]
		self.x_A = self.all_data[:,3]

		#self.x_A = 1 - 0.5*self.r*( self.x1_means + self.x2_means ) - ( 1 - self.r )*self.x1_means
		self.betas = 1 - self.phis/map( self.phi_R, self.x_A )
		#self.x_A = 1 - 0.5*self.r*( self.x1_means + self.x2_means ) - 1*self.x1_means

		self.final_x1 = self.x1_means[-1]
		self.final_x2 = self.x2_means[-1]

		self.final_x_A = self.x_A[-1]
		self.final_beta = self.betas[-1]

		self.final_pi = self.PI_bar( self.f_of_t[-1] )

		if self.printInfo:
			print "x_1,x2 = ", self.final_x1, self.final_x2

		if self.makeGraphs:

			pl.figure()
			pl.subplot(211)
			pl.plot( self.x1_means, label = "$x_1$" )
			pl.plot( self.x2_means, label = "$x_2$" )
			pl.legend()
			pl.xlabel("Steps")
			pl.ylabel("Frequency")

			pl.subplot(212)
			pl.plot( self.x_A, label = "$x_A$" )
			pl.plot( self.betas, label = "$\\beta$" )
			pl.legend()
			pl.xlabel("Steps")
			pl.ylabel("Frequency")

			if self.saveFigs:
				pl.savefig( self.fig_folder+"\\data%.2f_%.2f_%.2f.png"%(self.S,self.T,self.r) )

			##A picture of movement through triangle space
			pl.figure()
			pl.plot( [0,.5],[0,1], color = 'black', linewidth = 1 )
			pl.plot( [0.5,1],[1,0], color = 'black', linewidth = 1 )
			xa_s = np.linspace(0,1,101)
			pl.plot( xa_s, map(self.phi_R,xa_s), '--', color = 'black', linewidth = 2.5 )
			pl.xlabel( '$x_A$', fontsize = 30 )
			pl.ylabel( '$\\varphi$', fontsize = 30 )
			pl.plot( self.x_A, self.phis, 'o-', color = 'red', markersize = 3. )
			if self.saveFigs:
				pl.savefig( self.fig_folder+"\\triangle%.2f_%.2f_%.2f.png"%(self.S,self.T,self.r) )

			cmap = pl.get_cmap('Reds')
			pl.figure()
			pl.imshow( np.resize( self.f_of_t[-1], (self.steps,self.steps) ), origin = [0, 0], interpolation = 'nearest',\
				extent = [ 0,1,0,1], cmap = cmap)
			pl.xlabel('x2')
			pl.ylabel('x1')
			pl.colorbar()
			if self.saveFigs:
				pl.savefig( self.fig_folder+"\\heat%.2f_%.2f_%.2f.png"%(self.S,self.T,self.r) )

			if not self.saveFigs:
				pl.show()

def delta_x(i,j,M):
	"""This function is used to generate the triangle phase space diagrams.
	It returns the change in frequency at point i, j given a model m.
	
	Note
	====
	This was a mostly experimental idea, it is not currently in use

	"""

	f = np.zeros( (M.steps,M.steps) )
	f[i,j] = 1.
	try:
		f[i+1,j] = .01
	except IndexError:
		pass
	try:
		f[i-1,j] = .01
	except IndexError:
		pass
	try:	
		f[i,j+1] = .01
	except IndexError:
		pass
	try:
		f[i,j-1] = .01
	except IndexError:
		pass

	f = f/np.sum(f)
	#print f
	_,__,phi0,xA0 = M.return_data( f )
	#x10,x20 = np.sum(f,1).dot(M.xs), np.sum(f,0).dot(M.xs)
	df = M.df(f,0)
	df.resize((M.steps,M.steps))
	#dx1,dx2 = np.sum(df,1).dot(M.xs), np.sum(df,0).dot(M.xs)
	_,__,phi_1,xA_1 = M.return_data( f + df )

	return xA0, phi0, phi_1 - phi0, xA_1 - xA0

def triangle_vector_space(S,T,r, steps = 21):
	"""
	This draws the phase space in term of the triangular description of phase space, given a game
	specified by S, T and r. Steps defines the number ofdecrete steps to take in each axis.
	
	Note
	====
	This was a mostly experimental idea, it is not currently in use

	"""
	##Initialise a population at a certain point, with small frequencies of mutants around it

	M = GP_evo3(S,T,r=r, steps = steps)

	pl.figure()
	pl.plot( [0,.5],[0,1], color = 'black', linewidth = 1 )
	pl.plot( [0.5,1],[1,0], color = 'black', linewidth = 1 )
	xa_s = np.linspace(0,1,101)
	pl.plot( xa_s, map(M.phi_R,xa_s), '--', color = 'black', linewidth = 2.5 )
	pl.xlabel( '$x_A$', fontsize = 30 )
	pl.ylabel( '$\\varphi$', fontsize = 30 )
	for i in xrange(steps):
		for j in xrange(steps):

			arrow = delta_x(i,j,M)
			#print arrow
			pl.arrow( *arrow )

	pl.show()

def pi_SES(S,T):
	"""
	Returns the mean fitness of the population at the (unstructured) SES given values
	for S and T.

	"""
	if S + T > 2:
		return ( S + T )**2/( 4*( S + T - 1 ))
	else:
		return 1

##Sweeps through alpha for all models and plots final fitness for each. And also plots xbar and beta_bar for 
##model 3 only.
def make_figure(S,T, alphaPoints = 11, reps = 6):
	"""
	Makes the line figures of r vs pi used in the paper, as well as the plots of x and beta.
	Makes two plots on one figure given values for S and T.
	The top pannel is the equilibrium value for x and \beta plotted against r.
	The second figure plaots the equilibrium fitness as a fnction of r. It does this for the full model,
	the mixed strategy model, and the pure strategy model. It also plots three dashed lines for the all
	cooperate state, one for the unstructured SES and one for the structured SES.

	Inputs:
	S : float
		Parameter of the game
	T : float
		Parameter of the game
	alphaPoints : int {11}
		Number of points at which to sample alpha i.e. r
	reps : int {6}
		Number of times to repeat the whole thing and average over

	Returns:
		None

	Note
	====
		It will try to load the data from file if it exists, otherwise regenerate from scratch

	"""

	alphas = np.linspace(0,1,alphaPoints)

	##See if the data has already been generated
	try:
		
		f = open( """data_V3\\3 Mods\\fits_%.2f_%.2f.txt"""%(S,T),'r' )
		data = pickle.load(f)
		f.close()

		f = open( """data_V3\\3 Mods\\xs_%.2f_%.2f.txt"""%(S,T),'r' )
		mean_xs = pickle.load(f)
		f.close()

		f = open( """data_V3\\3 Mods\\betas_%.2f_%.2f.txt"""%(S,T),'r' )
		mean_betas = pickle.load(f)
		f.close()

		print "File Found"

	##If the file is not found
	except IOError:

		print "File not found, generating data"

		print S,T

		alphas = np.linspace(0,1,alphaPoints)
		mean_xs = np.zeros( (alphaPoints, reps) )
		mean_betas = np.zeros( (alphaPoints, reps) )

		data = np.zeros( (3,alphaPoints,reps) )
		for k in xrange(reps):
			print "Rep:",k
			for j,alpha in enumerate(alphas):

				# print alpha
				for modNum in [1,2,3]:

					model = GP_evo3( S, T, r = alpha, printInfo = False, makeGraphs = False, modelVersion = modNum)
					model.go()
					data[ modNum - 1, j, k] = model.final_pi

					if modNum == 3:
						mean_xs[j,k] = model.final_x_A
						mean_betas[j,k] = model.final_beta

		##Save the data in pickled files
		f = open( """data_V3\\3 Mods\\fits_%.2f_%.2f.txt"""%(S,T),'w' )
		pickle.dump( data, f )	
		f.close()

		f = open( """data_V3\\3 Mods\\xs_%.2f_%.2f.txt"""%(S,T),'w' )
		pickle.dump( mean_xs, f )	
		f.close()

		f = open( """data_V3\\3 Mods\\betas_%.2f_%.2f.txt"""%(S,T),'w' )
		pickle.dump( mean_betas, f )	
		f.close()

	pl.figure()

	mean_data = np.mean(data,2)
	pl.subplot(212)
	point_types = ['o-','*-','v-']
	for i in range(3):
		pl.plot(alphas, mean_data[i,:], point_types[i] )
	#pl.ylim([-1,1])
	pl.xlabel( 'r', fontsize = 30 )
	pl.ylabel( '$\\bar{\\pi}$', fontsize = 30 )
	pl.legend( ['Pure', 'Mixed', 'Full'], loc = 4)
	pl.plot( [0,1], [1,1], '--', color = 'blue' )
	if S + T > 2:
		pl.plot( [0,1], [pi_SES(S,T)]*2, '--', color = 'green')
		pl.plot( [0,1], [(S+T)/2.]*2, '--', color = 'red')
		#pl.ylim([-.1,( (S+T)/2. )*1.1])
	else:
		pass
		#pl.ylim([-.1,1.1])

	pl.subplot(211)
	pl.plot( alphas, np.mean(mean_xs,1) , 'o-', label = '$x_A$' )
	pl.plot( alphas, np.mean(mean_betas,1), '*-', label = '$\\bar{\\beta}$')
	#pl.xlabel( 'r', fontsize = 30 )
	pl.ylabel( '$\\bar{\\beta}$ / $x_A$', fontsize = 30 )
	pl.ylim([-1.1,1.1])
	if S+T>2:
		pl.legend(loc = 1)
	else:
		pl.legend(loc = 4)

	pl.savefig( "test_3_mods_%.3f_%.3f.eps"%(S,T) )

def examples(AP = 21, reps = 12):
	"""
	Makes a number of architypal figures by calling make_figure on several values of S and T.
	These are chossen to represent one DOL and one non-DOL game for the snowdrift, PD and Hamrnony
	game, as well as one non-DOL stag-hunt.

	Inputs
	======
	AP : int {21}
		Number of decrete points at which to sample alpha (i.e. r)
	reps : int {12}
		Number of times to repeat the process and mean over

	Returns
	=======
	None

	"""

	##A non-DOL Hamrnony Game
	for S,T in example_STs:
		make_figure(S,T, alphaPoints = AP, reps = reps)


def draw_ST_space(r = 0, points = 11, reps = 2):
	"""
	Sweeps over ST space, and makes two plots in one figure. These are the 
	value of x and \beta respectivly.

	Inputs
	======
	r : float [0,1] {0}
		Value of relatedness
	points : int {11}
		Number of decrete values off S and T to sample
	reps : int {2}
		Number of time to repeat the process

	"""

	Ss = np.linspace(-1,4,points)
	Ts = np.linspace(0,4,points)
	data_x_bar = np.zeros( ( points, points, reps ) )
	data_beta_bar = np.zeros( ( points, points, reps ) )
	for k in xrange(reps):
		print "Rep:", k + 1
		for i,S in enumerate( Ss ):
			for j,T in enumerate( Ts ):
				mod = GP_evo3(S,T, steps = 11, r = r, makeGraphs = False, printInfo = False)
				mod.go()
				data_x_bar[i,j,k] = mod.final_x_A
				data_beta_bar[i,j,k] = mod.final_beta

	f1 = open("""data_V3\\ST_sweep_beta%.2f.txt"""%r,'w')
	pickle.dump( data_beta_bar, f1 )
	f1.close()

	f2 = open("""data_V3\\ST_sweep_x%.2f.txt"""%r,'w')
	pickle.dump( data_x_bar, f2)
	f1.close()

	cmap = pl.get_cmap('Reds')
	pl.figure()
	pl.subplot(121)
	pl.imshow( np.mean( 1 - data_x_bar,2), origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
		extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0, vmax = 1, cmap = cmap)
	# pl.imshow( RGB_data, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
	#  	extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0, vmax = 1, cmap = cmap)
	pl.title('$\\bar{x}$')
	pl.xlabel('T')
	pl.ylabel('S')
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	pl.colorbar(shrink = .595)

	cmap = pl.get_cmap('seismic')
	pl.subplot(122)
	pl.imshow( np.mean( data_beta_bar,2), origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
		extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = -1, vmax = 1, cmap = cmap)
	# pl.imshow( RGB_data, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
	#  	extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0, vmax = 1, cmap = cmap)
	pl.title('$\\bar{\\beta}$')
	pl.xlabel('T')
	pl.ylabel('S')
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	pl.colorbar(shrink = .595)

	pl.savefig( "Figures\\V3\\r=%.2f.png"%r )

example_STs = [
				(0.4, 0.7),
				(1.8, 0.4),
				(-.4, 0.7),
				(0.4, 1.4),
				(-0.6, 1.7),
				(-0.2, 1.6)
				 ]

if __name__ == "__main__"

	##A few examples of things that you might use this script for

	##An example game
	S,T = -.2, 3.8
	M = GP_evo3(S,T)
	M.go()

	##Make the ST sweep figure for several values of r
	for r in np.linspace(0,1,6):
		draw_ST_space(r = r, points = 41, reps = 12)

	##Draw the triangle vector space
	S,T = example_STs[3]
	r = 1.
	triangle_vector_space(S,T,r)

	##More detailed output for all of the eaxmple games
	for S,T in example_STs:
		for r in np.linspace(0,1,4):

			M = GP_evo3(S,T, modelVersion = 3, steps = 11, r= r, saveFigs = True)
			M.go()

	##Run the examples as they are
	examples()

	##Make the r sweep figure

	S,T = -.2,4.
	make_figure(S,T,alphaPoints = 41, reps = 120)

	S,T = -.3,1.8
	make_figure(S,T,alphaPoints = 41, reps = 120)

