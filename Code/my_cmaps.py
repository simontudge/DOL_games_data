##Defiens some costom colour maps

import numpy as np
import pylab as pl

from matplotlib.colors import LinearSegmentedColormap

cdict1 = {'red':   ((0.0, 0.0, 0.0),
				    (0.5, 0.4, 0.4),
                    (1, .5, .5),
                   ),

         'green': ((0.0, 0.0, 0.0),
                   (1, 0., .0), 
                   ),

         'blue':   ((0.0, .5, .5),
         		    (0.5, 0.4, 0.4),
                    (1, 0.0, 0.0), 
                   ),

        }

cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (.5, 0.0, 0.0), 
                   (1, 1.0, 1.0), 
                   ),

         'green': ((0.0, 0.0, 0.0),
                   (1, 0., .0), 
                   ),

         'blue':   ((0.0, 1.0, 1.0),
                   (.5, 0.0, 0.0), 
                   (1, 0.0, 0.0), 
                   ),


        }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
blue_red2 = LinearSegmentedColormap('BlueRed2', cdict2)


# ##Some test data
# x = np.arange(0, np.pi, 0.1)
# y = np.arange(0, 2*np.pi, 0.1)
# X, Y = np.meshgrid(x,y)
# Z = np.cos(X) * np.sin(Y) * 10

# pl.figure()
# pl.imshow(Z, interpolation='nearest', cmap=blue_red1)
# pl.colorbar()