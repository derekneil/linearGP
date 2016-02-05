
# coding: utf-8

# In[1]:

from deap import base, creator, tools
import random
import numpy
import warnings
import sys
import time
from numpy import genfromtxt
import numpy as np
import warnings; 
with warnings.catch_warnings(): 
    warnings.simplefilter("ignore"); 
    import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

DEBUG = False

sizeOfIndividual = 25 #lines of Rx = Rx op Ry
populationSize = 40
mutationProb = 0.8
generations = 100
filename = 'iris_rescaled.csv'
selection = 'proportional'
replaced = populationSize

args = len(sys.argv)
if args >= 2:
    arg = sys.argv[1].lower().strip()
    if arg in 'help usage':
        print '\tusage: prgm.py [dataset.csv [generations]]'
    if '.csv' in arg:
        filename = arg
if args >= 3:
    try:
        arg = sys.argv[3]
        generations = int(arg)
    except:
        pass

dataset = genfromtxt(filename, delimiter=',')
outputRegisters = numpy.unique(dataset[:,numpy.size(dataset[0])-1:].flatten()).size


# In[2]:

# source: http://www.bhyuu.com/numpy-how-to-split-partition-a-dataset
# -array-into-training-and-test-datasets-for-e-g-cross-validation/
def get_train_test_inds(y,train=0.8):
    '''Generates indices, making random stratified split into training set 
    and testing sets with proportions 'train' and (1-train) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''
    y=numpy.array(y)
    train_inds = numpy.zeros(len(y),dtype=bool)
    test_inds = numpy.zeros(len(y),dtype=bool)
    values = numpy.unique(y)
    for value in values:
        value_inds = numpy.nonzero(y==value)[0]
        numpy.random.shuffle(value_inds)
        n = int(train*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds, test_inds

train_inds, test_inds = get_train_test_inds(dataset[:,-1:].flatten())
train = dataset[train_inds]
test  = dataset[test_inds]


# In[3]:

def trainVsTest(hof):
    print 'train  |  test'
    testHof = creator.Individual(hof[0].copy())
    testHof.fitness.values = toolbox.evaluate(testHof, test)
    show(hof[0], testHof)

def show(old, new=None): #debugging/output worker function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if new != None: #print side by side individuals
            buffer = ' ' * (len(old[0])+len(old[0])-1+2)
            for i in range(max(len(old),len(new))):
                a , b = buffer, buffer
                try: a = old[i]
                except: pass
                try: b = new[i]
                except: pass
                print a, b , '<---' if a!=buffer and b!=buffer and (a != b).any() else ''
            try:
                fit = old.fitness.values
                prettyFitness = '\t'.join(["{:.4f}".format(x) for x in fit])
                print 'left ',prettyFitness
            except:
                pass
            try:
                fit = new.fitness.values
                prettyFitness = '\t'.join(["{:.4f}".format(x) for x in fit])
                print 'right', prettyFitness
            except:
                pass
        else: #print one or more single individual(s)
            for individual in old:
                try:
                    fit = individual.fitness.values
                    prettyFitness = '\t'.join(["{:.4f}".format(x) for x in fit])
                    print individual, '\t', prettyFitness
                except:
                    print individual
            try:
                fit = old.fitness.values
                prettyFitness = '\t'.join(["{:.4f}".format(x) for x in fit])
                print prettyFitness
            except:
                pass

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# In[4]:

fitnessMeasures = 2 #tp and fp per class
creator.create("FitnessMulti", base.Fitness, weights=(numpy.concatenate((numpy.ones(fitnessMeasures/2), numpy.negative(numpy.ones(fitnessMeasures/2))), axis=0).tolist()))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMulti)

minAttributeIndex = 0 # lowest value in zero indexed array of values
maxRegistryIndex = numpy.shape(dataset)[1] - 2 + outputRegisters # = numInputAttributes - target, and zero indexed + outputRegisters
maxOperatorIndex = 3 # op is {0,1,2,3} mul,add,sub,div

def initIndividual(cr8tr, sizeOfIndividual):
    return cr8tr([random.randint(minAttributeIndex, maxRegistryIndex), random.randint(minAttributeIndex, maxOperatorIndex), random.randint(minAttributeIndex, maxRegistryIndex)] for _ in range(sizeOfIndividual))

toolbox = base.Toolbox()
toolbox.register("individual", initIndividual, creator.Individual, sizeOfIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# In[5]:

def removeIntrons(individual):
    reducedIndividual = individual.copy()
    
    #TODO implement intron filtering and count real program length
    
    return reducedIndividual

def evaluate(individual, dataset=train):
    result  = numpy.zeros(fitnessMeasures*outputRegisters)
    results = numpy.zeros(fitnessMeasures*outputRegisters)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        reducedIndividual = removeIntrons(individual)
        
        for row in dataset:
            r = numpy.concatenate((numpy.zeros(outputRegisters), row), axis=0)
            target = row[-1] - 1
            for x, op, y in reducedIndividual:
                if   op == 0: r[x] = r[x] * r[y]
                elif op == 1: r[x] = r[x] + r[y]
                elif op == 2: r[x] = r[x] - r[y]
                elif op == 3:
                    try:      r[x] = r[x] / r[y] 
                    except: continue #leave r[x] alone if divide by zero

            #TODO use unique correct prediction ratio and program length as only two fitness measures
            
            predicted = numpy.argmax(r[:outputRegisters]) - 1
            
            result[predicted] += 1 if predicted == target else 0
            results[target] += 1
            result[predicted+outputRegisters] += 1 if predicted != target else 0
            results[target+outputRegisters] += 1
            
    fitnessPerClass = result/results
    return numpy.mean(fitnessPerClass.reshape(-1, outputRegisters), axis=1).tolist() #must return iterable tuple

def mutateMatrix(ind, low, up, indpb, up1=None):
    global mutations
    if up1==None: up1 = up
    
    for row in ind:
        if random.random() < indpb:
            for i in range(len(row)):
                if random.random() < indpb:
                    mutations += 1
                    #TODO increment to next register/operator instead of randomly picking a new one
                    if i==1: row[i] = random.randint(low, up1) #op
                    else:    row[i] = random.randint(low, up)  #Rx, Ry
    return ind,

toolbox.register("mutate", mutateMatrix, low=minAttributeIndex, up=maxRegistryIndex, indpb=mutationProb, up1=maxOperatorIndex)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate)


# In[6]:

# build population and do initial evaluation
pop = toolbox.population(n=populationSize)
for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
    ind.fitness.values = fit

# track best individual
hof = tools.HallOfFame(1, similar=numpy.array_equal)
hof.update(pop)

mutations = 0
lastHof = None


# In[7]:

def evolve(generation, pop, hof, lastHof):
    if DEBUG: popcopy = [c.copy() for c in pop]
    
    offspring = map(toolbox.clone, toolbox.select(pop, k=replaced))
    random.shuffle(offspring)

    for mutant in offspring:
        if random.random() < mutationProb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pop[:] = offspring
    
    if DEBUG:
        print 'population diff'
        for i in range(len(pop)):
            show(popcopy[i],pop[i])
            print'----------------'
        
    hof.update(pop)
    
    if hof[0].fitness.values[0] > 0.6:
        if lastHof==None:
            lastHof = hof[0]
            if DEBUG:
                trainVsTest(hof)
                print 'generation',generation
        elif hof[0].fitness.values[0] > lastHof.fitness.values[0]:
            lastHof = hof[0]
            if DEBUG:
                trainVsTest(hof)
                print 'generation',generation
    
    return pop, hof, lastHof


# In[8]:

fig, ax = plt.subplots(frameon=False)
ax.set_xlim(-0.1,1.1), ax.set_ylim(-0.1,1.1)
ax.set_xlabel('True Positive')
ax.set_ylabel('False Positive')
scat = ax.scatter(x=[], y=[], s=1, lw=5)

BLK = (0, 0, 0, 1)
BLU = (0, 0, 1, 1)
GRN = (0, 1, 0, 1)
CYN = (0, 1, 1, 1)
RED = (1, 0, 0, 1)
MGT = (1, 0, 1, 1)
YEL = (1, 1, 0, 1)

startTime = time.time()
data = np.zeros(populationSize, dtype=[('fitnessValues', float, 2), ('color', float, 4)])
def update(generation):
    global pop, hof, lastHof
    pop, hof, lastHof = evolve(generation, pop, hof, lastHof)
    
    for i, ind in zip(range(populationSize), pop):
        data['fitnessValues'][i] = ind.fitness.values
        prod = data['fitnessValues'][i][0] * data['fitnessValues'][i][1]
        color = BLK
        if prod>0.6:
            color = RED
        elif prod<0.05:
            color = MGT
        data['color'][i] = color
    
    scat.set_offsets(    data['fitnessValues'] )
    scat.set_edgecolors( data['color'] )

    ax.set_title('Fitness generation: %6d / %d \t\truntime: %.2f seconds' %(generation+1,generations, time.time()-startTime))

animation = FuncAnimation(fig, update, generations, interval=10, repeat=False)
plt.show()


# In[9]:

print "\ndataset\tpop\tmuPb\tgens\tselection\tre\tmutations"
print filename, '\t', populationSize,  '\t', mutationProb, '\t',     generations,'\t', selection, '\t', replaced,'\t', mutations


# In[10]:

print '\nHALL OF FAME'
print 'size of best individual:', len(hof[0])

trainVsTest(hof)


# In[ ]:



