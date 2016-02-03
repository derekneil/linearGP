
# coding: utf-8

# In[1]:

from deap import base, creator, tools
import random
import numpy
import warnings
import sys
import time
from numpy import genfromtxt

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
        print '\tusage: prgm.py [dataset.csv [tourGenerations]]'
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
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions 'train' and (1-train) of initial sample.
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

creator.create("FitnessMulti", base.Fitness, weights=(numpy.concatenate((numpy.ones(outputRegisters+1), numpy.negative(numpy.ones(outputRegisters))), axis=0).tolist()))
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

def evaluate(individual, dataset=train):
    result  = numpy.zeros(outputRegisters*2+1) # *2 for tp and fp +1 for overall accuracy
    results = numpy.zeros(outputRegisters*2+1) # 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        #TODO implement intron filtering and count real program length
        
        for row in dataset:
            r = numpy.concatenate((numpy.zeros(outputRegisters), row), axis=0)
            target = row[-1]
            for x, op, y in individual:
                if   op == 0: r[x] = r[x] * r[y]
                elif op == 1: r[x] = r[x] + r[y]
                elif op == 2: r[x] = r[x] - r[y]
                elif op == 3:
                    try:      r[x] = r[x] / r[y] 
                    except: continue #leave r[x] alone if divide by zero

            #TODO use unique correct prediction ratio and program length as only two fitness measures
            
            predicted = numpy.argmax(r[:outputRegisters])+1
            result[0]  += 1 if predicted == target else 0
            results[0] += 1
            result[predicted] += 1 if predicted == target else 0
            results[target] += 1
            result[predicted+outputRegisters] += 1 if predicted != target else 0
            results[target+outputRegisters] += 1
        
    return (result/results).tolist() #must return iterable tuple

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
    
    offspring = toolbox.select(pop, k=replaced)
    random.shuffle(offspring)

    # Apply mutation on some of the offspring
    for mutant in offspring:
        if random.random() < mutationProb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate modified offspring (the individuals with an invalid(deleted) fitness)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
            
    if DEBUG:
        print 'population diff'
        for i in range(len(pop)):
            show(popcopy[i],pop[i])
            print'----------------'
        
    hof.update(pop)
    
    if hof[0].fitness.values[0] > 0.6:
        if lastHof==None:
            lastHof = hof[0]
            trainVsTest(hof)
            print 'generation',generation
        elif hof[0].fitness.values[0] > lastHof.fitness.values[0]:
            lastHof = hof[0]
            trainVsTest(hof)
            print 'generation',generation
    
    return pop, hof, lastHof


# In[ ]:

startTime = time.time()
for generation in range(generations):
    pop, hof, lastHof = evolve(generation, pop, hof, lastHof)

endTime = time.time()
runTime = (endTime - startTime)

print "\ndataset\tpop\tmuPb\tgens\tselection\tre\texec time\tmutations"
print filename, '\t', populationSize,  '\t', mutationProb, '\t', generations,     '\t', selection, '\t', replaced,'%0.2f' % runTime, '\t', mutations


# In[ ]:

print '\nHALL OF FAME'
print 'size of best individual:', len(hof[0])

trainVsTest(hof)


# In[ ]:



