import numpy as np
import warnings; 
with warnings.catch_warnings(): 
    warnings.simplefilter("ignore"); 
    import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

generations    = 100
populationSize = 50

BLK = (0, 0, 0, 1)
BLU = (0, 0, 1, 1)
GRN = (0, 1, 0, 1)
CYN = (0, 1, 1, 1)
RED = (1, 0, 0, 1)
MGT = (1, 0, 1, 1)
YEL = (1, 1, 0, 1)


fig, ax = plt.subplots(frameon=False)
ax.set_xlim(0,1), ax.set_ylim(0,1)
ax.set_xlabel('F1')
ax.set_ylabel('F2')
scat = ax.scatter(x=[], y=[], s=1, lw=5)

def getData():                # F1                          #F2
    return [np.random.uniform(0, 0.25), np.random.uniform(0.75, 1)]


data = np.zeros(populationSize, dtype=[('fitnessValues', float, 2), ('color', float, 4)])
def update(g):

    #get your real data here
    data['fitnessValues'] = getData()
    
    #adjust color based on data
    for i, d in zip(range(populationSize), data['fitnessValues']):
        prod = d[0] * d[1]
        color = BLK
        if prod>0.6:
            color = RED
        elif prod<0.05:
            color = MGT
        data['color'][i] = color
    
    scat.set_offsets(    data['fitnessValues'] )
    scat.set_edgecolors( data['color'] )

    ax.set_title('Fitness gen: %6d / %d' %(g,generations) )


animation = FuncAnimation(fig, update, generations, interval=100, repeat=False)
plt.show()
