"""
A script and collection of functions for drawing the ESSs of the parental agent model.
Rather than integrate numerically this version just plots the pre-calculated ESSs.

See Also
========
parental_agent :
	Does the same as this but using a numerical approach

"""


import numpy as np
import pylab as pl

import parental_agent as PA

def f(p,q,S,T):
	"""
	The fitness the an individual playing mixed strategy p receives on playng an individual with mixed strategy
	q, for a game given by S and T.

	"""
	return ( 1 - p )*( 1 - q ) + ( 1 - p )*q*S + p*( 1 - q )*T

def pi_SES(S,T):
	"""
	The fitness at the (unstructured) SES.

	"""
	return ( S + T )**2/( 4*( S + T - 1 ))

def ESS_pure(r,S,T):
	"""The Predicted ESS in terms of frequecy of strategy B for an assorted population of mixed strategy individuals.
	Inputs:
	r : relatedness
	S
	T

	"""

	x = ( ( 1 - r )*T - 1 )/( ( 1 - r )*( S + T - 1 ) )

	if x > 1:
		x = 1
	elif x < 0:
		x = 0
	
	if S + T >= 1:
		return x
	else:
		if x > 0.5:
			return 0
		else:
			return 1

def fit_pure( x, r, S, T ):
	"""
	The fitness obtained for a population given by state x for relatedness r and
	game given by S and T.

	"""
	return (-1 + x)*(-1 + (-1 + r)*(-1 + S + T)*x)


def ESS_mixed(r,S,T):
	"""
	The ESS of the mixed game, given relatedness r, and a game given by S and T.

	"""

	if r == 0:
		return ESS_pure(r,S,T)
	else:
		x = (-1 + r*(-1 + S) + T)/((1 + r)*(-1 + S + T))
		if x > 1:
			x = 1
		elif x < 0:
			x = 0
		
		if S + T >= 1:
			return x
		else:
			if x > 0.5:
				return 0
			else:
				return 1

def ESS_full(r,S,T,*args,**kwargs):
	"""
	Returns the ESS of the full game, this is simply a thin wrapper for the class ESS_class
	Takes r,S and T.
	Passes args and kwargs to ESS_class/
	"""

	mod = ESS_class(r,S,T,*args,**kwargs)
	return mod.getESS()

class ESS_class:
	"""
	Class for determinign the ESS of the full parental model.
	Call getESS to retun the ESS of the game.
	"""

	def __init__(self,r,S,T, debug = False):
		"""
		init for ESS_class
		
		Inputs
		=======
		Takes r, S and T
		Debug bool {False} :
			Set to true to print verbous listing of why a certain ESS has been found.

		"""
		self.r, self.S, self.T = r,S,T
		self.debug = debug

	def f(self,p,q):
		"""
		Fitness for miced strategy p playing mixed strategy q.

		"""
		return ( 1 - p )*( 1 - q ) + ( 1 - p )*q*self.S + p*( 1 - q )*self.T

	def pi_self(self,x,y):
		"""
		Fitness an agent receives on playing itself

		"""
		return 0.5*( self.f(x,y) + self.f(y,x) )

	def pi(self,x,y,r,x_bar):
		"""
		Fitness for an agent given by strategy (x,y) with relatedness r, given mean value of 
		x in the population given by x_bar.
		"""
		return r*self.pi_self(x,y) + ( 1 - r )*self.f(x,x_bar)

	##The average fitness for a certain population state
	def pi_bar(self,state):
		"""
		The average fitness for the population described by state.

		"""

		a,b,c,d = state
		x_bar = c + d
		return a*self.pi(0,0,self.r,x_bar)\
		+b*self.pi(0,1,self.r,x_bar)\
		+c*self.pi(1,0,self.r,x_bar)\
		+d*self.pi(1,1,self.r,x_bar)

	
	def xA( self, state, r ):
		"""
		Average phenotyic value of A for a certain state and given r.

		"""
		a,b,c,d = state
		return 1 - 0.5*r*( c + 2*d + b ) - ( 1 - r )*( c + d )


	def beta(self, state, r):
		"""
		Average phenotyic value of A for a certain state and given r.

		"""
		a,b,c,d = state
		##The frequency of unlike interactions
		Ex1 = c + d
		Ex2 = b + d
		E_x1_x2 = d
		phi = r*( Ex1 + Ex2 - 2*E_x1_x2 ) + 2*( 1 - self.r )*( Ex1 - Ex1**2 )
		
		xa = self.xA(state,r)
		if xa == 0 or xa == 1:
			return 0
		##The frequency of unlike interaction that would be expected if all interactions were random
		phiR = 2*xa*(1-xa)

		return 1 - phi/phiR

	def getESS(self):
		"""
		Returns the ESS of this particular setup.

		"""

		if self.r == 0:

			xs = ( ( 1 - self.r )*self.T - 1 )/( ( 1 - self.r )*( self.S + self.T - 1 ) )

			if self.T +self.S > 1:
				if xs <= 0:
					self.ESS_type = "Pure_AA"
					return ( 0.5, 0.5, 0, 0 )
				elif xs >= 1:
					self.ESS_type = "Pure_BB"
					return (0,0, .5, .5 )
				else:
					self.ESS_type = "AA_BB_stable"
					return (0.5*(1-xs),0.5*(1-xs), .5*xs, .5*xs )
 
			else:

				if xs >= 1:
					self.ESS_type = "Pure_AA"
					return ( 0.5, 0.5, 0, 0 )
				elif xs <= 0:
					self.ESS_type = "Pure_BB"
					return (0,0, .5, .5 )
				else:
					self.ESS_type = "AA_BB_bistable"
					if xs > 0.5:
						return ( 0.5, 0.5, 0, 0 )
					else:
						return ( 0, 0, 0.5, 0.5 )
 

		if self.r == 1 and self.S + self.T > 2:
			self.ESS_type = "AB_BA_neutral"
			return (0,.5,.5,0)

		##Strategies 01 and 10 are equal fitness
		if self.S + self.T != 1:
			xs = (-1 + self.T)/(-1 + self.S + self.T)
		else:
			xs = (-1 + self.T)/(-1 + self.S + self.T + 0.00001 )
		##Check that this is in the domain of x
		if 0 <= xs <= 1:
			if self.debug:
				print "01 and 10 have equal fitness %.3f"%xs
			payoff = self.pi(0,1,self.r,xs)
			p00 = self.pi(0,0,self.r,xs)
			p11 = self.pi(1,1,self.r,xs)
			##Check that these strategies are the fitest in this case.
			if payoff >= p00 and payoff >= p11:
				##We've found an ESS
				self.ESS_type = "AB_BA_stable"
				return (0,1-xs,xs,0)
			else:
				if self.debug:
					print "But no ESS"

		##mix of 00 and 11
		if self.S +self.T != 1:
			xs = (1 + (-1 + self.r)*self.T)/((-1 + self.r)*(-1 + self.S + self.T))
		else:
			xs = (1 + (-1 + self.r)*self.T)/((-1 + self.r)*(-1 + self.S + self.T + 0.00001))
		##Check that this is in the domain of x
		if 0 <= xs <= 1:
			if self.debug:
				print "00 and 11 have equal fitness at %.3f"%xs
			payoff = self.pi(0,0,self.r,xs)
			p10 = self.pi(1,0,self.r,xs)
			p01 = self.pi(0,1,self.r,xs)
			##Check that these strategies are the fitest in this case.
			if payoff >= p10 and payoff >= p01:
				##We've found an ESS
				##Check for stability of this fixed point. This comes from the gradient of pi[0,0]-pi[1,0]
				if self.S + self.T >= 1:	
					self.ESS_type = "AA_BB_stable"
					return (1-xs,0,0,xs)
				else:
					self.ESS_type = "AA_BB_bistable"
					if xs > .5:
						return (1,0,0,0)
					else:
						return (0,0,0,1)
					
			else:
				if self.debug:
					print "But no ESS"

		##mix of 00 and 10
		if self.S +self.T != 1:
			xs = (2 - self.r*self.S + (-2 + self.r)*self.T)/(2.*(-1 + self.r)*(-1 + self.S + self.T))
		else:
			xs = (2 - self.r*self.S + (-2 + self.r)*self.T)/(2.*(-1 + self.r)*(-1 + self.S + self.T + 0.0001 ))
		##Check that this is in the domain of x
		if 0 <= xs <= 1:
			if self.debug:
				print "00 and 10 have equal fitness %.3f"%xs
			payoff = self.pi(0,0,self.r,xs)
			p11 = self.pi(1,1,self.r,xs)
			p01 = self.pi(0,1,self.r,xs)
			##Check that these strategies are the fittest in this case.
			if payoff >= p11 and payoff >= p01:
				##We've found an ESS
				##Check for stability of this fixed point. This comes from the gradient of pi[0,0]-pi[1,0]
				if self.S + self.T >= 1:
					self.ESS_type = "AA_BA_stable"
					return (1-xs,0,xs,0)
				else:
					self.ESS_type = "AA_BA_bistable"
					if xs > .5:
						return (1,0,0,0)
					else:
						return (0,0,1,0)
				
			else:
				if self.debug:
					print "But no ESS"

		##mix of 01 and 11
		if self.S +self.T != 1:
			xs = (2 - 2*self.T + self.r*(-2 + self.S + 3*self.T))/(2.*(-1 + self.r)*(-1 + self.S + self.T))
		else:
			xs = (2 - 2*self.T + self.r*(-2 + self.S + 3*self.T))/(2.*(-1 + self.r)*(-1 + self.S + self.T + 0.0001 ))
		##Check that this is in the domain of x
		if 0 <= xs <= 1:
			if self.debug:
				print "01 and 11 have equal fitness %.3f"%xs
			payoff = self.pi(0,1,self.r,xs)
			p00 = self.pi(0,0,self.r,xs)
			p10 = self.pi(1,0,self.r,xs)
			##Check that these strategies are the fitest in this case.
			if payoff >= p00 and payoff >= p10:
				##We've found an ESS
				self.ESS_type = "AB_BB_stable"
				return (0,1-xs,0,xs)
			else:
				if self.debug:
					print "But no ESS"

		if self.debug:
			print "No Mixed ESS Found"
		##We need to check all fitness and find a maximum
		# p00 = self.pi(0,0,self.r,1)
		# p01 = self.pi(0,1,self.r,1)
		# p10 = self.pi(1,0,self.r,1)
		# p11 = self.pi(1,1,self.r,1)
		p00 = self.pi(0,0,self.r,.5)
		p01 = self.pi(0,1,self.r,.5)
		p10 = self.pi(1,0,self.r,.5)
		p11 = self.pi(1,1,self.r,.5)
		fs = [p00,p01,p10,p11]
		if self.debug:
			print fs
		i = fs.index( max(fs) )
		e = [0,0,0,0]
		e[i] = 1
		##Set the type of the ESS as a pure type of the given index
		self.ESS_type = "Pure_"+['AA','AB','BA','BB'][i]
		return e


class ESS_lines:
	"""
	Class for drawing a set of lines delimiting the various ESSs.
	Makes a plot of ST space and draws lines delimiting various regions
	of different ESSs.

	"""

	def __init__(self,r):
		"""
		Inti function, take a value of relatedness r
		"""
		self.r = r


	#These are the equations for various lines that may or may not show up depending on r...
	def line1( self, T ):
		return  self.r / ( self.r - 1 )

	def line2( self, T ):
		return ( 2*self.r - self.r*T  )/( 3*self.r - 2)

	def line3(self,T):
		return 0

	def line4(self,T):
		return 2 - T

	def line5(self,T):
		return - T

	def line6(self,T):
		return (2-2*T + self.r*T)/self.r

	def pline1(self,T):
		x =	max( self.line1(T), self.line2(T) )
		if x > 0:
			return None
		else:
			return x


	def pline2(self,T):
		return max( self.line4(T), self.line3(T) )

	def pline3(self,T):
		return min( self.line1(T), self.line5(T) )

	def pline4(self,T):
		return min( self.line4(T), self.line6(T) )

	def plot_all(self):

		##Linewidth
		lw = 4
		#pl.figure()
		Ts = np.linspace(0,3,1001)
		if self.r == 0:

			pl.plot( [ Ts[0], Ts[-1]], [ 0,0 ], '-' , linewidth =lw, color='black' )
			pl.plot( [ 1, 1 ],[ -1, 2 ],'-', linewidth =lw, color='black' )

		elif self.r == 1:
				pl.plot( [ 0, 3 ],[ 2, -1 ],'-', linewidth =lw,color='black' )

		else:

			pl.plot( Ts, map(self.pline1,Ts) ,'-',  linewidth =lw, color = 'black')
			pl.plot( Ts, map(self.pline2,Ts) ,'-',  linewidth =lw,color = 'black')
			pl.plot( Ts, map(self.pline3,Ts) ,'-',  linewidth =lw, color = 'black')
			pl.plot( Ts, map(self.pline4,Ts) ,'-',  linewidth =lw, color = 'black')
			pl.plot( [1,1],[1,3],'-',  linewidth =lw, color = 'black' )

		pl.xlabel('T', fontsize = 24)
		pl.ylabel('S', fontsize = 24)
		pl.ylim(-1,2)
		#pl.show()

def maximal_possible_fitness(S,T):
	"""
	Returns the maximal possible fitness for S and T.

	"""
	if S + T < 2:
		return 1
	else:
		return (S+T)/2.


##A dictionary of all ESS types together with an integer identifying it
ESS_dict = {"Pure_AA":0, "Pure_AB":1, "Pure_BA":2, "Pure_BB":3,\
		"AA_BA_stable": 4, "AA_BB_stable": 5, "AB_BA_stable": 6, "AB_BB_stable": 7,\
		"AA_BA_bistable": 8, "AA_BB_bistable": 9,\
		"AB_BA_neutral":10}

##A reverse version of the previous dictionary
ESS_dict_rev = { ESS_dict[k]:k for k in ESS_dict.iterkeys() }

def ESS_type(r,S,T):
	"""Returns the qualitative type of the ESS"""
	mod = ESS_class(r,S,T)
	mod.getESS()
	return mod.ESS_type

def test( S, T, steps = 11, *args, **kwargs):
	"""
	Quickly comapers the agent and the ESS versions of the models by making a figure to be eyeballed.
	The figure is the density of the four key genotypes vs. r

	"""

	rs = np.linspace(0,1,steps)
	from parental_agent import GP_evo3 as GP
	x1s = []
	x2s = []
	xT = []
	yT = []
	for r in rs:

		mod = GP( S, T, r = r, makeGraphs = False, *args, **kwargs)
		mod.go()
		x1,x2 = mod.final_x1,mod.final_x2
		x1s.append( x1 )
		x2s.append( x2 )

		f = ESS_full(r,S,T)
		xT.append( f[2] + f[3] )
		yT.append( f[1] + f[3] )

	pl.figure()
	pl.plot(rs,x1s,'o',color = 'blue')
	pl.plot(rs,x2s,'o',color = 'green')

	pl.plot(rs,xT)
	pl.plot(rs,yT)
	pl.xlabel("r")
	pl.ylabel("x")
	pl.ylim([-0.1,1.1])

	pl.show()

def fit_figure(S,T,steps = 101, agent = False):#
	"""
	Makes a plot of fitness against r for a given game specified via S and T.
	Does this by comparing all three versions of the model.
	inputs:
	=======
	S : float
		Parameter of the game
	T : float
		Parameter of the game
	steps : int {101}
		Number of decrete points at which to sample r
	agent : bool {False}
		Whether or not to add points for the agent, will take significantly longer if this
		is true.

	"""

	fitsFull = []
	fitsPure = []
	fitsMixed = []
	rs = np.linspace(0,1,steps)
	for r in rs:

		##Fitness for the Pure model
		x = ESS_pure(r,S,T)
		fitsPure.append( fit_pure(x,r,S,T) )

		##Fitness for the mixed model
		x = ESS_mixed(r,S,T)
		fitsMixed.append( f(x,x,S,T) )

		##Fitness for the full Model
		mod = ESS_class(r,S,T)
		state = mod.getESS()
		fitsFull.append( mod.pi_bar(state) )

	if agent:
		rs_agent = np.linspace(0,1,11)
		fitFull_agent = []
		fitMixed_agent = []
		fitPure_agent = []
		for r in rs_agent:

			M = PA.GP_evo3(S,T, r = r, makeGraphs = False)
			M.go()
			fitFull_agent.append( M.final_pi )

			M = PA.GP_evo3(S,T, r = r, modelVersion = 2, makeGraphs = False)
			M.go()
			fitMixed_agent.append( M.final_pi )

			M = PA.GP_evo3(S,T, r = r, modelVersion = 1, makeGraphs = False)
			M.go()
			fitPure_agent.append( M.final_pi )

	#pl.figure()
	pl.plot(rs, fitsFull, color = 'red', label = 'Full' )
	pl.plot(rs,fitsMixed, color = 'green', label = 'Mixed')
	pl.plot(rs,fitsPure,color = 'blue',label = 'Pure')
	pl.plot([0,1],[1]*2,'--',color = 'blue')
	if agent:
		pl.plot(rs_agent, fitFull_agent, 'o', color = 'red' )
		pl.plot(rs_agent,fitMixed_agent, 'o', color = 'green')
		pl.plot(rs_agent,fitPure_agent, 'o',color = 'blue')
	if S + T > 2:
		pl.plot( [0,1], [pi_SES(S,T)]*2, '--', color = 'green')
		pl.plot( [0,1], [(S+T)/2.]*2, '--', color = 'red')
	pl.xlabel('r')
	pl.ylabel("$\\bar{\\pi}$", fontsize = 24)
	if S+T>2:
		pl.yticks( np.linspace( -0, 2., 5 ) )
		pl.ylim([-0.1,2.])
	else:
		pl.yticks( np.linspace( -0.2, 1.2, 5 ) )
		pl.ylim([-0.2,1.1])

	pl.legend( loc = 2, prop={'size':16}  )
	#pl.show()


def xA_beta(S,T,steps=101, agent = False):
	"""
	Makes a plot of xa and beta vs. r for the full model.
	inputs
	======
	S : float
		Parameter of the game
	T : float
		Parameter of the game
	steps : int {101}
		Integer number of steps at which to sample r
	agent : bool {False}
		Whether or not to include points for the agent based model

	"""
	
	rs = np.linspace(0,1,steps)
	xAs = []
	betas = []
	for i,r in enumerate( rs ):

		mod = ESS_class( r, S, T )
		x = mod.getESS()
		xAs.append( mod.xA(x,r) )
		betas.append( mod.beta(x,r) )

	if agent:
		xAs_agent = []
		betas_agent = []
		rs_agent = np.linspace(0,1,11)
		for r in rs_agent:

			M = PA.GP_evo3(S,T, r = r, makeGraphs = False)
			M.go()
			xAs_agent.append( M.final_x_A )
			betas_agent.append( M.final_beta )

	#pl.figure()
	pl.plot(rs,xAs, label = "$x_A$", color = 'blue')
	pl.plot(rs,betas, label = "$Cov(p,p')$", color = 'red')
	#pl.xlabel('r')
	pl.ylabel("$x_A$ $and$ $Cov(p,p')$", fontsize = 18)
	if S+T>2:
		pl.yticks( np.linspace( -1., .6, 5 ) )
		pl.ylim([-1.,.6])
		pl.legend( loc = 3, prop={'size':18} )
	else:
		pl.yticks( np.linspace( -.4, 1.2, 5 ) )
		pl.ylim([-.4,1.2])
		pl.legend( loc = 2, prop={'size':18} )

	if agent:
		pl.plot(rs_agent,xAs_agent, 'o', label = "$x_A$", color = 'blue' )
		pl.plot(rs_agent,betas_agent, 'o', label = "$Cov(p,p')$", color = 'red' )
	#pl.show()

##Sweeps through ST space and draws fitness, beta and xa respectively
def ST_pi(r,steps = 99, graph = True):
	"""
	Makes a heat map over ST space for a given value of r, using the full model, plotting
	fitness.

	Inputs
	======
	r : float
		Value of relatedness
	steps : int {99}
		Number of points to sample S and T at.
	graph : bool {True}
		Whether or not to make the graph

	Returns
	=======
		numpy.array of data

	"""

	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			mod = ESS_class(r,S,T)
			state = mod.getESS()
			dataFull[i,j] = mod.pi_bar(state)/maximal_possible_fitness(S,T)

	#pl.figure()
	if graph:
		cmap = pl.get_cmap('Reds')
		pl.imshow(dataFull, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
				extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], cmap = cmap, vmin = 0, vmax = .5*(Ss[-1]+Ts[-1]))
		pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
		pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
		pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
		#pl.colorbar()
		#pl.show()

	return dataFull

def ST_beta(r,steps = 99):
	"""
	Makes a heat map of beta over ST space, for a given value of r.
	Uses the full model.
	Steps : int {99}
		Number of steps to sample S and T at
	Returns
	=======
	None
	"""

	Ss = np.linspace(-1,3,steps)
	Ts = np.linspace(0,4,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			mod = ESS_class(r,S,T)
			state = mod.getESS()
			dataFull[i,j] = mod.beta(state,r)

	#pl.figure()
	cmap = pl.get_cmap('coolwarm')
	pl.imshow(dataFull, origin = [Ts[0], Ss[0]],\
			extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = -1,vmax = 1, cmap = cmap)
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	#pl.colorbar()
	#pl.show()

def ST_xA(r,steps = 99):
	"""
	Makes a heat map for x over ST space for a given r using the full model.
	inputs
	======
	r : float
		Value of relatedness
	steps: int {99}
		Number of decrete points at which to sample S and T

	"""


	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			mod = ESS_class(r,S,T)
			state = mod.getESS()
			dataFull[i,j] = mod.xA(state,r)

	#pl.figure()
	cmap = pl.get_cmap('Reds')
	pl.imshow(dataFull, origin = [Ts[0], Ss[0]],\
			extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], vmin = 0,vmax = 1, cmap = cmap)
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	#pl.title("r=%.3f"%r)
	#pl.colorbar()

	#pl.show()

def ST_pi_pure(r,steps = 99):
	"""
	Makes a heat map over ST space for a given value of r, using the pure strategy model, plotting
	fitness.

	Inputs
	======
	r : float
		Value of relatedness
	steps : int {99}
		Number of points to sample S and T at.

	"""

	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			x = ESS_pure(r,S,T)
			dataFull[i,j] = fit_pure(x,r,S,T)/maximal_possible_fitness(S,T)

	#pl.figure()
	cmap = pl.get_cmap('Reds')
	pl.imshow(dataFull, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
			extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], cmap = cmap, vmin = 0, vmax = .5*(Ss[-1]+Ts[-1]))
	pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
	#pl.colorbar()
	#pl.show()

def ST_pi_mixed(r,steps = 99, graph = True):
	"""
	Makes a heat map over ST space for a given value of r, using the miced strategy model, plotting
	fitness.

	Inputs
	======
	r : float
		Value of relatedness
	steps : int {99}
		Number of points to sample S and T at.
	graph : bool {True}
		Whether or not to make the graph

	Returns
	=======
		numpy.array of data

	"""

	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	dataFull = np.zeros( (steps,steps) )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			x = ESS_mixed(r,S,T)
			dataFull[i,j] = f(x,x,S,T)/maximal_possible_fitness(S,T)

	#pl.figure()
	if graph:

		cmap = pl.get_cmap('Reds')
		pl.imshow(dataFull, origin = [Ts[0], Ss[0]], interpolation = 'nearest',\
				extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], cmap = cmap, vmin = 0, vmax = .5*(Ss[-1]+Ts[-1]))
		pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
		pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
		pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )
		#pl.colorbar()
		#pl.show()

	return dataFull

def relative_fitness(r,steps = 99):
	"""
	Makes a heat map over ST space for a given value of r, using the full model, plotting the relative 
	fitness of the full model vs. the mixed strategy model. i.e. full_fit - mixed_fit.

	Inputs
	======
	r : float
		Value of relatedness
	steps : int {99}
		Number of points to sample S and T at.
	graph : bool {True}
		Whether or not to make the graph

	Returns
	=======
		numpy.array of data

	"""

	d_full = ST_pi(r,steps, graph = False)
	d_mixed = ST_pi_mixed(r,steps, graph = False)
	data = d_full - d_mixed

	Max = max( np.max(data), 0.5)

	Ss = np.linspace(-1,2,steps)
	Ts = np.linspace(0,3,steps)
	cmap = pl.get_cmap('seismic')
	pl.imshow(data, origin = [Ts[0], Ss[0]],\
			extent = [ Ts[0],Ts[-1],Ss[0],Ss[-1] ], aspect = 1/1.2, cmap = cmap, vmin = -1.0, vmax = 1.0)
	pl.xticks([0,1,2,3,4])
	pl.yticks([-1,0,1,2,3])
	#pl.plot( [ Ts[0], Ts[-1] ],[ 0,0 ],color='black', linewidth = 2.5 )
	#pl.plot( [ 1, 1 ],[ Ss[0],Ss[-1] ],'--',color='black',  linewidth = 2.5 )
	#pl.plot( [ 0, 3 ],[ 2, -1 ],'--',color='black',  linewidth = 2.5 )

	return data

def t_sweep(S=-0.5,r=0, T_steps = 101):
	"""
	Sweeps through values fo T for a fixed S. Making a line plot with x and beta on it.
	Also adds some usful annotations.

	Inputs
	======
	S : float {-0.5}
		Parameter of the game
	r : float {0}
		Value for the relatedness
	T_steps : int {101}
		Number of decrete points at which to sample T
	Notes
	=====
	Very much used for experimental probing

	"""

	data = np.zeros( ( T_steps, 2 ) )
	Ts = np.linspace(0,4,T_steps)
	for i,T in enumerate(Ts):
		mod = ESS_class( r, S, T )
		x = mod.getESS()
		data[i,0] = mod.xA(x,r)
		data[i,1] = mod.beta(x,r)

	##Possible tipping point
	tp1 =  (-2 + r*S)/(r-2)
	tp2 =  (2*r + 2*S - 3*r*S)/r
	#pl.figure()
	pl.plot( Ts, data )
	pl.xlabel("T")
	##Line marking the end of the staghunt game
	pl.plot([1,1],[-1.1,1.1],'--',color = 'black')
	##Line marking the beginning of the DOL region
	pl.plot( [2-S,2-S], [-1.1,1.1], '--', color = 'black')
	##Theoretical transition point
	if tp1 > 1:
		pl.plot([tp1]*2,[-1.1,1.1],'--',color = 'Red')
	if tp2 > 1:
		pl.plot([tp2]*2,[-1.1,1.1],'--',color = 'Red')	
	pl.ylabel("Frequency")
	pl.legend(["$x_A$","$\\beta$"])
	pl.ylim(-1.05,1.05)
	#pl.show()

##Note this isn't really the final figure, this has been changed! What a stupid name, what were you thinking
def final_figure(n= 3):
	"""Make the final figure. For n (default = 3) values of r plot along the first row relative fitness.
	That is fitness of the full model minus (or over) fitness in the mixed case.
	The second row is Xa and the 3rd row is beta. """
	rs = np.linspace(0,1,n)
	pl.figure()
	for i,r in enumerate(rs):

		pl.subplot( 3, n, i+1)
		relative_fitness(r = r, steps = 200)
		if i == n - 1 and n <= 3:
			pl.colorbar( ticks = np.linspace(-1.2,1.2,5))
		pl.locator_params( nbins=4)

		pl.subplot( 3, n, n + i + 1)
		ST_xA(r=r,steps = 200)
		if i ==n-1 and n <=3:
			pl.colorbar( ticks = np.linspace(0,1,5))
		if i ==0:
			pl.ylabel("S", fontsize = 20)
		pl.locator_params( nbins=4)

		pl.subplot( 3, n, 2*n + i + 1)
		ST_beta(r=r,steps = 200)	
		if i ==n-1 and n <= 3:
			pl.colorbar( ticks = np.linspace(-1,1,5) )	
		if i ==n/2:
			pl.xlabel("T", fontsize = 20)
		pl.locator_params( nbins=4)

	pl.savefig("figures\\final_heatmap_%d.png"%n)


def ESS_map(r = 0, steps = 200, sMin = -1, sMax = 3, tMin = 0, tMax = 4):
	"""
	Over the whole of ST space plots the qualitative type of the ESS
	If regions is True then it creates an additional map divided into qualitatively different regions.
	Note this isn't really used, except for experimenting, the output is very ugly!
	
	Inputs
	======
	r : float {0}
		Value for relatedness
	steps : int {200}
		decrete points at which to sample both S and T
	sMin : float {-1}
		Mimimum value for S
	sMax : float {3}
		Maximum value for S
	tMin : float {0}
		Mimimum value for T
	tMax : float {4}
		Maximum value for T

	"""

	Ss = np.linspace(sMin,sMax,steps)
	Ts = np.linspace(tMin,tMax,steps)
	data = np.zeros( (steps,steps), int )
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):
			ET = ESS_type(r,S,T)
			data[i,j] = ESS_dict[ET]


	fig, ax = pl.subplots()
	cax = ax.imshow(data, origin = [tMin, sMin], interpolation = 'nearest',\
			extent = [ tMin,tMax,sMin,sMax ], vmin = 0, vmax = 10)
	cbar = fig.colorbar(cax, ticks = range(11) )
	cbar.ax.set_yticklabels( [ ESS_dict_rev[i] for i in range(11)] )

	pl.plot( [ tMin, tMax ],[ 0,0 ],color='black', linewidth = 2.5 )
	pl.plot( [ 1, 1 ],[ sMin,sMax ],'--',color='black',  linewidth = 2.5 )
	pl.plot( Ts, map( lambda x: 2 - x, Ts),'--',color='black',  linewidth = 2.5 )
	pl.ylim(sMin, sMax)

	#pl.savefig("Figures\\ESS_space\\ESS_map_%.2f.png"%r)

	#pl.show()
	# return data

##At the moment this plots an alternative version of the final figure, which
##Shows the fitness of each of the models for increasing values of r, relative
##to the theroetically maximum value. That is max - actual.
if __name__ == "__main__":
	pl.figure()
	steps = 3
	for i,r in enumerate( np.linspace(0,1,steps) ):

		pl.subplot( 3, steps, i + 1 )
		pl.title("r=%.2f"%r)
		ST_pi_pure(r)
		pl.xticks( np.linspace( 0, 4, 5) )
		pl.yticks( np.linspace( -1, 3, 5) )
		pl.xlim( [0,3] )
		pl.ylim([-1,2])

		pl.subplot(3,steps, steps + i + 1)
		#pl.title("mixed r=%.2f"%r)
		ST_pi_mixed(r)
		pl.xticks( np.linspace( 0, 4, 5) )
		pl.yticks( np.linspace( -1, 3, 5) )
		pl.xlim( [0,3] )
		pl.ylim([-1,2])

		pl.subplot( 3, steps, 2*steps + i + 1 )
		#pl.title("mixed r=%.2f"%r)
		pl.xticks( np.linspace( 0, 4, 5) )
		pl.yticks( np.linspace( -1, 3, 5) )
		ST_pi(r)
		pl.xlim( [0,3] )
		pl.ylim([-1,2])

	pl.show()
	


# #pl.show()
# #pl.annotate()
# steps = 21

# for i,r in enumerate( [.3,0.7,1.] ):

# 	#pl.subplot(3,1,i+1)
# 	pl.figure()

# 	lines = ESS_lines(r)
# 	lines.plot_all()

# 	#pl.subplot(311)
# 	d = relative_fitness( r, steps )
# 	pl.ylim(-1,2)
# 	pl.xlim(0,3)

# 	if r == 0.3:
# 		pl.annotate("[1,+1]",[0.4,0.4], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,-1]",[0.6,1.65], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,-1] and [0,-1]",[1.8,1.2], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,+1]\nor [0,+1]",[0.35,-.85], fontsize = 15, ha = 'center',fontweight = 'bold')
# 		pl.annotate("[1,+1]\nand [0,-1]",[1.4,.35], fontsize = 15, ha = 'center', rotation = -45,fontweight = 'bold')
# 		pl.annotate("[1,+1]\nor [0,-1]",[1.0,-.6], fontsize = 15, ha = 'center', rotation = -40, fontweight = 'bold')		
# 		pl.annotate("[0,-1]",[2.1,-.65], fontsize = 15,fontweight = 'bold')
# 	elif r==0.7:
# 		pl.annotate("[1,-1]",[0.57,1.63], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,-1] and [0,-1]",[1.65,1.1], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,+1]",[0.55,-0.], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[0,-1]",[2.4,-0.5], fontsize = 15,fontweight = 'bold')
# 		pl.annotate("[1,+1]\nand [0,-1]",[1.77,0.01], rotation =-57, fontsize = 15,fontweight = 'bold', ha = 'center')
# 	elif r == 1.:
# 		pl.annotate("[1,-1] and [0,-1]\nNeutrally Stable",[1.8,1.2], rotation = -40,fontsize = 15,fontweight = 'bold', ha = 'center')
# 		pl.annotate("[1,+1]",[1.,0], rotation = -40,fontsize = 15,fontweight = 'bold', ha = 'center')

# pl.show()

# pl.subplot(312)
# d = relative_fitness( .7, steps )

# pl.subplot(313)
# d = relative_fitness( 1., steps )


# r = 1/3.
# S = -0.5
# T = 1.25
# print ESS_type(r,S,T)
# pl.figure()
# fit_figure(S,T)
# pl.show()

# test(S,T)

# S,T = -.2, 3.8
# pl.figure()
# pl.subplot(211)
# xA_beta( S, T, agent = True )
# pl.subplot(212)
# fit_figure( S, T, agent = True )
# pl.show()

#ESS_map(r=0.8, sMin = -10, tMin = -2, tMax = 7)

# M = ESS_lines(r=0.5)
# M.plot_all()

#test(0.5,1.6,t_f = 5000)

# rs = np.linspace(0,1,21)
# for r in rs:
# 	ESS_map(r=r)

# final_figure()




# pl.figure()
# for i,r in enumerate( np.linspace(0,1,6) ):
# 	print r
# 	pl.subplot(2,3,i+1)
# 	ST_xA(r=r)

# pl.figure()
# for i,r in enumerate( np.linspace(0,1,6) ):
# 	print r
# 	pl.subplot(2,3,i+1)
# 	ST_beta(r=r)

# pl.figure()
# for i,r in enumerate( np.linspace(0,1,6) ):
# 	print r
# 	pl.subplot(2,3,i+1)
# 	ST_pi(r=r)

# pl.show()

# S,T = .1,1.6
# test(S,T, makeGraphs = False, steps = 41, t_f = 5000)

# rs = np.linspace(0,1,8)
# pl.figure()
# for i,r in enumerate(rs):
# 	pl.subplot(2,4,i+1)
# 	t_sweep(S = -0.1,r = r)

# pl.show()
# for n in [3,4,6,8]:
# 	final_figure(n)