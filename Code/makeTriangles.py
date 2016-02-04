##makeTriangles.py

##Do the make triangle thing again.

##Fitness as a function of state space pi = x_a + 1/2*(S+T-1)*rho

import numpy as np
import pylab as pl

def pi(xa, phi):
	
	if phi <= 2*min([ xa, 1 - xa ]):
		return xa + (1/2.)*(S+T-1)*phi
	else:
		return np.nan


def ESS(S,T):
	return S/( S + T - 1 )

def phi_R(x):
	return 2*x*(1-x)

def SOF(S,T):

	return ( S + T )/( 2*( S + T - 1 ) )

def stability( x, phi ):

	if phi <= 2*min([ x, 1 - x ]):
		return abs( (phi*(-1 + S) + (2 - phi*(-1 + S + T))*x - 2*x**2)/2. )
		#return (phi*(-1 + S) + (2 - phi*(-1 + S + T))*x - 2*x**2)/2. 
	else:
		return np.nan

def stableLine(x):
	return -2*(-x + x**2)/(1 - S - x + S*x + T*x)

S = .8
T = 2.9

points = 501

xa_s = np.linspace( 0, 1, points )
phis = np.linspace( 0, 1, points )

pis = np.array( [ [ pi(x,phi)  for phi in phis ] for x in xa_s ] )


cmap = pl.get_cmap('OrRd')
pl.figure()
pl.subplot(121)
pl.imshow( pis.transpose(), origin = [0,0], extent = [0,1,0,1], cmap = cmap)
pl.plot( xa_s, map(phi_R,xa_s), '--', color = 'black', linewidth = 2.5 )
pl.plot( [0,.5],[0,1], color = 'black', linewidth = 5 )
pl.plot( [0.5,1],[1,0], color = 'black', linewidth = 5 )
pl.xlabel( '$x_A$', fontsize = 30 )
pl.ylabel( '$\\varphi$', fontsize = 30 )
pl.title( '''$\\bar{\\pi}$''', fontsize = 30, y=1.05 )
pl.locator_params(nbins=4)

arrowWidth = 0.001
##Anotate three key points
pl.annotate('B', fontsize = 25, xy=(.5, 1), xytext=(.16, .75),
            arrowprops=dict(facecolor='black', shrink=0.05, linewidth = arrowWidth),
            )

ses = SOF(S,T)
pl.annotate('A', fontsize = 25, xy=( ses , phi_R( ses )), xytext=(.75,.68),
            arrowprops=dict(facecolor='black', shrink=0.05, linewidth = arrowWidth),
            )

# pl.annotate('All C', fontsize = 30, xy=(1, 0), xytext=(.2, .8),
#             arrowprops=dict(facecolor='black', shrink=0.05, linewidth = arrowWidth),
#             )

pl.colorbar(shrink  = .47, ticks = [0,.6,1.2,1.8] )

ST = np.array( [ [ stability(x,phi)  for phi in phis ] for x in xa_s ] )
pl.subplot(122)
pl.plot( xa_s, map(phi_R,xa_s), '--', color = 'black', linewidth = 2.5 )
pl.plot( xa_s, map(stableLine,xa_s), '-.', color = 'black', linewidth = 2.5 )
pl.plot( [0,.5],[0,1], color = 'black', linewidth = 5 )
pl.plot( [0.5,1],[1,0], color = 'black', linewidth = 5 )
pl.xlabel( '$x_A$', fontsize = 30 )
#pl.ylabel( '$\\varphi$', fontsize = 30 )
pl.locator_params(nbins=4)
pl.imshow( ST.transpose(), origin = [0,0], extent = [0,1,0,1], cmap = cmap)

ess = ESS(S,T)
pl.annotate('ESS', fontsize = 25, xy=( ess , phi_R(ess) ), xytext=(.05,.75),
            arrowprops=dict(facecolor='black', shrink=0.05, linewidth = arrowWidth),
            )

pl.title( '''$|\\pi_{A}-\\bar{\\pi}|$''', fontsize = 30,  y=1.05 )
pl.colorbar(shrink  = .47, ticks = [0,.2,.4] )

pl.show()

##pl.savefig("triangles2.png")
#pl.show()
