import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

generations    = 100
populationSize = 50

fig, ax = plt.subplots(frameon=False)
ax.set_xlim(0,1), ax.set_ylim(0,1)

BLK = (0, 0, 0, 1)
BLU = (0, 0, 1, 1)
GRN = (0, 1, 0, 1)
CYN = (0, 1, 1, 1)
RED = (1, 0, 0, 1)
MGT = (1, 0, 1, 1)
YEL = (1, 1, 0, 1)

data = np.zeros(populationSize, dtype=[('fitnessValues', float, 2), ('color', float, 4)])

scat = ax.scatter(x=data['fitnessValues'][:,0], y=data['fitnessValues'][:,1], \
                  s=1, lw=5, edgecolors=data['color'])

def update(g):

    #get your real data here
    data['fitnessValues'] = np.random.uniform(0, 1, (populationSize, 2))
    
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

animation = FuncAnimation(fig, update, generations, interval=1000, repeat=False)
plt.show()
