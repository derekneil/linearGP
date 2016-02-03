
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

initialSizeOfIndividual = 25 #lines of Rx = Rx op Ry
maxSizeOfIndividual = 250
populationSize = 40
crossoverProb = 0.8
mutationProb = 0.8
generations = 9000
filename = 'iris_rescaled.csv'
selection = 'tournament'
replaced = 2
# selection = 'proportional'

args = len(sys.argv)
if args >= 2:
    arg = sys.argv[1].lower().strip()
    if arg in 'help usage':
        print '\tusage: prgm.py [tournament/proportional [dataset.csv [tourGenerations]]]'
    if arg in 'tournament proportional':
        selection = arg
if args >= 3:
    arg = sys.argv[2].lower().strip()
    if '.csv' in arg:
        filename = arg
if args >= 4:
    try:
        arg = sys.argv[3]
        generations = int(arg)
    except:
        pass
            
if selection in 'proportional':
    selection = 'proportional'
    generations = int(float(replaced)/populationSize * generations)
    replaced = populationSize
    
else:
    selection = 'tournament'

fullDataset = genfromtxt(filename, delimiter=',')
outputRegisters = numpy.unique(fullDataset[:,numpy.size(fullDataset[0])-1:].flatten()).size


# In[2]:

# source: http://www.bhyuu.com/numpy-how-to-split-partition-a-dataset-array-into-training-and-test-datasets-for-e-g-cross-validation/
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

train_inds, test_inds = get_train_test_inds(fullDataset[:,-1:].flatten())
train = fullDataset[train_inds]
test  = fullDataset[test_inds]

#use global dataset variable to hold either training or test data
dataset = train[::]


# In[3]:

def trainVsTest(hof):
    print 'train  |  test'
    testHof = creator.Individual(hof[0].copy())
    testHof.fitness.values = toolbox.evaluate(testHof, test)
    show(hof[0], testHof)

def show(old, new=None): #debugging/output worker function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if new != None:
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
        else:
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


# In[5]:

minAttributeIndex = 0 # lowest value in zero indexed array of values
maxRegistryIndex = numpy.shape(dataset)[1] - 2 + outputRegisters # = numInputAttributes - target, and zero indexed + outputRegisters
maxOperatorIndex = 3 # op is {0,1,2,3} mul,add,sub,div

def initIndividual(cr8tr, initialSizeOfIndividual):
    return cr8tr([random.randint(minAttributeIndex, maxRegistryIndex), random.randint(minAttributeIndex, maxOperatorIndex), random.randint(minAttributeIndex, maxRegistryIndex)] for _ in range(initialSizeOfIndividual))

toolbox = base.Toolbox()
toolbox.register("individual", initIndividual, creator.Individual, initialSizeOfIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# In[6]:

def evaluate(individual, dataset=train):
    result = numpy.zeros(outputRegisters*2+1) # +1 for overall accuracy
    results = numpy.zeros(outputRegisters*2+1) # *2 for tp and fp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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

            predicted = numpy.argmax(r[:outputRegisters])+1
            result[0]  += 1 if predicted == target else 0
            results[0] += 1
            result[predicted] += 1 if predicted == target else 0
            results[target] += 1
            result[predicted+outputRegisters] += 1 if predicted != target else 0
            results[target+outputRegisters] += 1
        
#     if DEBUG: print result, ' / ', results
    
    return (result/results).tolist() #must return iterable tuple

def cxTwoPointMatrix(ind1, ind2):
    #exchange lines of instructions between ind1 and ind2
    #based on http://deap.readthedocs.org/en/master/examples/ga_onemax_numpy.html
    
    pt1, pt2 = [random.randint(0, len(ind1)-1) for _ in range(2)]
    if pt2 == pt1:
        pt2 += 1
    elif pt2 < pt1: # Swap the two cx points
        pt1, pt2 = pt2, pt1
        
    if DEBUG: print pt1, ':', pt2
    ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2].copy(), ind1[pt1:pt2].copy()
    result = ind1, ind2
    
    # crossing over different lengths of individuals led to programatic issues with numpy.ndarrays
#     pt3, pt4 = [random.randint(0, len(ind2)-1) for _ in range(2)]
#     if pt4 == pt3:
#         pt4 += 1
#     elif pt4 < pt3: # Swap the two cx points
#         pt3, pt4 = pt4, pt3

#     if DEBUG: print pt1, ':', pt2, '<->',pt3, ':', pt4
    
#     chunk1, chunk2 = ind1[pt1:pt2].copy(), ind2[pt3:pt4].copy()
        
#     mask = numpy.ones(len(ind1), dtype=bool)
#     mask[pt1:pt2] = False
#     newind1 = ind1[mask].copy() #lose reference to object

#     mask = numpy.ones(len(ind2), dtype=bool)
#     mask[pt3:pt4] = False
#     newind2 = ind2[mask].copy() #lose reference to object
    
#     newind1 = numpy.insert(newind1, pt1, chunk2, 0)
#     newind2 = numpy.insert(newind2, pt3, chunk1, 0)
    
#     newind1 = creator.Individual(newind1)
#     newind2 = creator.Individual(newind2)
    
#     result = newind1 if len(newind1) <= maxSizeOfIndividual else ind1, \
#         newind2 if len(newind2) <= maxSizeOfIndividual else ind2
    
    return result

def mutateMatrix(ind, low, up, indpb, up1=None):
    if up1==None: up1 = up
    for row in ind:
        if random.random() < indpb:
            for i in range(len(row)):
                if random.random() < indpb:
                    if i==1: row[i] = random.randint(low, up1) #op
                    else:    row[i] = random.randint(low, up)  #Rx, Ry
    return ind,

toolbox.register("mate", cxTwoPointMatrix)
toolbox.register("mutate", mutateMatrix, low=minAttributeIndex, up=maxRegistryIndex, indpb=mutationProb, up1=maxOperatorIndex)
if selection is 'tournament':
    toolbox.register("select", tools.selTournament, tournsize=populationSize)
elif selection is 'proportional':
    toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate)


# In[7]:

# build population and do initial evaluation
pop = toolbox.population(n=populationSize)
for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
    ind.fitness.values = fit

# track best individual
hof = tools.HallOfFame(1, similar=numpy.array_equal)
hof.update(pop)

print 'dataset\tpop\tcxPb\tmuPb\tgens\tselection\tre'
print filename, '\t', populationSize, '\t', crossoverProb, '\t', mutationProb, '\t', generations, '\t', selection, '\t', replaced


# In[ ]:

crossovers = 0
mutations = 0
startTime = time.time()
lastHof = None
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
for g in range(generations):

    offspring = None
    modified  = None
    if selection is 'tournament':
        offspring = toolbox.select(pop, k=4)
        modified  = map(toolbox.clone, tools.selBest(offspring, k=replaced))
    elif selection is 'proportional':
        offspring = map(toolbox.clone, toolbox.select(pop, k=replaced))
        random.shuffle(offspring)
        modified = offspring

    # Apply crossover on some of the offspring
#     newmodified = [] #required when new individuals created from different length crossovers
    for child1, child2 in zip(modified[::2], modified[1::2]):
        if random.random() < crossoverProb:
            crossovers +=2
            child1, child2 = toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
#         newmodified.append(child1)
#         newmodified.append(child2)
        
#     modified = newmodified

    # Apply mutation on some of the offspring
    for mutant in modified:
        if random.random() < mutationProb:
            mutations +=1
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    # Evaluate modified offspring (the individuals with an invalid(deleted) fitness)
    invalid_ind = [ind for ind in modified if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Add offspring back into population based on selection policy
    if DEBUG: popcopy = [c.copy() for c in pop]
    
    if selection is 'tournament':
        worst = tools.selWorst(offspring, 2)
        worst[0][:] = modified[0][:]
        worst[1][:] = modified[1][:]
            
    elif selection is 'proportional':
        pop[:] = offspring
            
    if DEBUG:
        print 'population diff'
        for i in range(len(pop)):
            show(popcopy[i],pop[i])
            print'----------------'
        
    hof.update(pop)
    
    if (hof[0].fitness.values[0] > 0.6 and hof[0] is not lastHof):
        lastHof = hof[0]
        trainVsTest(hof)
        print 'generation',g
    
    if not is_interactive(): print CURSOR_UP_ONE + ERASE_LINE, 'generation:',g

# end for loop
endTime = time.time()
runTime = (endTime - startTime)

print "\nexec time\tcrossovers\tmutations"
print '%0.2f' % runTime, '\t', crossovers,'\t', mutations,'\t'


# In[ ]:

print '\nHALL OF FAME'
print 'size of best individual:', len(hof[0])

trainVsTest(hof)


# In[ ]:



