
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
import multiprocessing
firstImport = True


# In[2]:

DEBUG = False

sizeOfIndividual = 25 #lines of Rx = Rx op Ry
populationSize = 2 if DEBUG else 40
mutationProb = 0.95
generations = 10 if DEBUG else 100
filename = 'tic-tac-toe_decimal.csv'
# filename = 'iris_rescaled.csv'
dominance = 'RANK'
# dominance = 'COUNT'

processes = 4
interval = 2000 if DEBUG else 1

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if not is_interactive():
    args = len(sys.argv)
    if args >= 2:
        arg = sys.argv[1].lower().strip()
        if arg in 'help usage':
            print '\tusage: prgm.py [dataset.csv [generations [RANK/COUNT]]]'
        if '.csv' in arg:
            filename = arg
    if args >= 3:
        try:
            arg = sys.argv[3]
            generations = int(arg)
        except:
            pass
    if args >= 4:
        arg = sys.argv[4]
        if arg in 'COUNT':
            dominance = 'COUNT'
        elif arg in 'RANK':
            dominance = 'RANK'

dataset = genfromtxt(filename, delimiter=',')
outputRegisters = numpy.unique(dataset[:,numpy.size(dataset[0])-1:].flatten()).size


# In[3]:

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


# In[4]:

def trainVsTest(ind):
    ind = creator.Individual(removeIntrons(ind))
    ind.fitness.values = toolbox.evaluate(ind, test)
    print 'size of best individual:', len(ind)
    print 'train  |  test'
    testHof = creator.Individual(ind.copy())
    testHof.fitness.values = toolbox.evaluate(testHof, test)
    show(ind, testHof)

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


# In[5]:

fitnessMeasures = 2 #tp and fp per class
creator.create("FitnessMulti", base.Fitness,                weights=(numpy.concatenate((numpy.ones(fitnessMeasures/2),                                           numpy.negative(numpy.ones(fitnessMeasures/2))),                                          axis=0).tolist()))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMulti)

minAttributeIndex = 0 # lowest value in zero indexed array of values
maxRegistryIndex = numpy.shape(dataset)[1] - 2 + outputRegisters # = row - (outpus + zero indexed) + outputRegisters
maxOperatorIndex = 3 # op is {0,1,2,3} mul,add,sub,div

def initIndividual(cr8tr, sizeOfIndividual):
    return cr8tr([random.randint(minAttributeIndex, maxRegistryIndex),                   random.randint(minAttributeIndex, maxOperatorIndex),                   random.randint(minAttributeIndex, maxRegistryIndex)] for _ in range(sizeOfIndividual))

toolbox = base.Toolbox()

pool = multiprocessing.Pool(processes)
#weird issue where this never works on first run of code, only on subsequent runs..??
if firstImport:
    firstImport = False
    processes = 1
else: 
    toolbox.register("map", pool.map)
    processes = 4

toolbox.register("individual", initIndividual, creator.Individual, sizeOfIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# In[6]:

def removeIntrons(individual):
    variables = set(range(outputRegisters))
    
    reducedIndividual = []
    for x,op,y in reversed(individual):
        if x in variables:
            variables.add(y)
            reducedIndividual.insert(0, [x,op,y])
    
    return reducedIndividual

def evaluate(individual, dataset=train):
    result  = numpy.zeros(fitnessMeasures*outputRegisters)
    results = numpy.zeros(fitnessMeasures*outputRegisters)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        reducedIndividual = removeIntrons(individual)
        
        for row in dataset:
            r = numpy.concatenate((numpy.negative(numpy.ones(outputRegisters)), row), axis=0)

            for x, op, y in reducedIndividual:
                if   op == 0: r[x] = r[x] * r[y]
                elif op == 1: r[x] = r[x] + r[y]
                elif op == 2: r[x] = r[x] - r[y]
                elif op == 3:
                    try:      r[x] = r[x] / r[y] 
                    except: continue #leave r[x] alone if divide by zero

            target = row[-1] - 1
            predicted = numpy.argmax(r[:outputRegisters])
            result[predicted] += 1 if predicted == target else 0
            results[target] += 1
            result[predicted+outputRegisters] += 1 if predicted != target else 0
            results[target+outputRegisters] += 1
    
    fitnessPerClass = result/results
    individualFitness = numpy.mean(fitnessPerClass.reshape(-1, outputRegisters), axis=1).tolist()
    if DEBUG: print 'fitnessPerClass', fitnessPerClass
    if DEBUG: print 'individualFitness', individualFitness
    return individualFitness #must return iterable tuple

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


# In[7]:

pop = toolbox.population(n=populationSize)
for ind, fit in zip(pop, toolbox.map(toolbox.evaluate, pop)):
    ind.fitness.values = fit

# track best individual across generations
hof = tools.HallOfFame(1, similar=numpy.array_equal)
hof.update(pop)

mutations = 0
lastHof = None


# In[8]:

def ndEQ(a,b): #required for comparing matrix individuals with DEAP's ParetoFront hallOfFrame
    return numpy.all((a==b).flatten())

def getDominanceValues(pop):
    paretoDominance = numpy.zeros(shape=len(pop),dtype=int)
    for i, a in enumerate(pop):
        for j, b in enumerate(pop):
            if i!=j:
                if dominance in 'RANK':
                    paretoDominance[i] += 1 if b.fitness.dominates(a.fitness) else 0
                elif dominance in 'COUNT':
                    paretoDominance[i] += 1 if a.fitness.dominates(b.fitness) else 0
                else:
                    raise NameError('you idiot')
    return paretoDominance

def getTopPareto(pop, offspring, n):
    pop.extend(offspring)
    
    pareto = getDominanceValues(pop)
    
    indicies = numpy.argsort(pareto)[:n]
    if dominance in 'COUNT':
        indicies = numpy.argsort(pareto)[::-1][:n]
    if DEBUG: print 'sorted pareto',dominance, pareto[indicies]
        
    newPop = []
    for i in indicies:
        newPop.append(pop[i])
        
    return newPop

def evolve(generation, pop, hof, lastHof):
    if DEBUG: popcopy = toolbox.map(toolbox.clone, pop)
    
    offspring = toolbox.map(toolbox.clone, pop)

    for mutant in offspring:
        if random.random() < mutationProb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pop[:] = getTopPareto(pop, offspring, populationSize)
    
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
                trainVsTest(hof[0])
                print 'generation',generation
        elif hof[0].fitness.values[0] > lastHof.fitness.values[0]:
            lastHof = hof[0]
            if DEBUG:
                trainVsTest(hof[0])
                print 'generation',generation
    
    return pop, hof, lastHof


# In[ ]:

fig, ax = plt.subplots(frameon=False)
ax.set_xlim(-0.1,1.1), ax.set_ylim(-0.1,1.1)
ax.set_xlabel('True Positive')
ax.set_ylabel('False Positive')
scat = ax.scatter(x=[], y=[], s=1, lw=5)

BLK = (0, 0, 0, 1)
RED = (1, 0, 0, 1)

paretoFront = tools.ParetoFront(ndEQ)
startTime = time.time()
def update(generation):
    global pop, hof, lastHof
    pop, hof, lastHof = evolve(generation, pop, hof, lastHof)
    
    paretoFront.update(pop)
    paretoFrontSize = len(paretoFront)
    
    data = np.zeros(populationSize+paretoFrontSize, dtype=[('coordinates', float, 2), ('color', float, 4)])
    for i, ind in enumerate(pop):
        data['coordinates'][i] = ind.fitness.values
        data['color'][i] = BLK
    
    for i, ind in enumerate(paretoFront):
        data['coordinates'][populationSize+i] = ind.fitness.values
        data['color'][populationSize+i] = RED
    
    scat.set_offsets(    data['coordinates'] )
    scat.set_edgecolors( data['color'] )

    ax.set_title('data:'+filename[:5]+' Pareto '+dominance+' front: %d/%d on %d processes\ngen: %3d /%d  mu(%.2f): %d in %.2f seconds'                  %(paretoFrontSize, populationSize, processes, generation+1, generations, mutationProb, mutations, time.time()-startTime))

animation = FuncAnimation(fig, update, generations, interval=interval, repeat=False)
plt.show()

print '\nHALL OF FAME'

trainVsTest(hof[0])


# In[ ]:



