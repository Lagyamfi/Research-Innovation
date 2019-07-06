from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter

#Parameter used for running main simulation of algorithm
pop_size = 100
dim = 32
generations = 1000
p_mutation = 0.3
p_crossover = 0.8
tournament_size = 7

# Class to define population
class Population():

    # Class to define individual solution.
    class Individual():

        def __init__(self, dim):
            self.dim = dim
            self.axis = np.arange(dim)
            self.values = np.random.permutation(self.axis)

        def fitness(self):
            # Calculates the fitness counting the number of attacks between queens.
            diag1 , diag2 = self.values + self.axis , self.values - self.axis
            counter = 0
            for i in range(0,self.dim-1):
                for j in range(i+1,self.dim):
                    if diag1[i] == diag1[j]: counter += 1
                    if diag2[i] == diag2[j]: counter += 1
            return counter

        def mutation(self, positions):
            #derangement of specific positions in self.values
            visited_positions = []
            for i in positions[:-1]:
                visited_positions += i
                j = np.random.choice([k for k in positions if k not in visited_positions], 1)
                self.values[j], self.values[i] = self.values[i], self.values[j]

    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.axis = np.arange(dim)
        self.individuals = [ self.Individual(dim) for _ in range(n) ]

    def fitness(self):
        # Calculates fitness for all individuals in population, sorts fitness and individuals by fitness, and returns fitness.
        fitness = [ind.fitness() for ind in self.individuals]
        self.individuals = [element[0] for element in sorted( zip(self.individuals, fitness), key = lambda x: x[1])]
        return sorted(fitness)

    def cx_crossover(self, ind1, ind2):
        # Swaps elements from son1 and son2 untill completing a cycle on ind1 and ind2, starting in a random position.
        son1, son2 = deepcopy(ind1), deepcopy(ind2)
        start = np.random.choice(self.axis, 1)
        i = start
        while True:
            son1.values[i], son2.values[i] = son2.values[i], son1.values[i]
            i = np.where(ind1.values == ind2.values[i])
            if i == start: break
        return son1, son2

    def flat_mutation_selection(self, p):
        # Selection algorithm for mutation with random probability
        prob = [(1-p)/2, (1-p)/2] + [p*i for i in [(1/2)**(s+1) for s in range(self.dim-3)] + [(1/2)**(self.dim-3)] ]
        num_mutations = [ np.random.choice(np.arange(self.dim) , 1, p = prob ) for i in range(self.n) ]
        return [ np.random.choice(self.axis, n) for n in num_mutations ]


    def tournament_crossover_selection(self, tournament_size):
        #Algorithm for selections of parents for crossover based on tournament size
        fitness = self.fitness()
        positions = [ sorted(np.random.choice(np.arange(self.n), tournament_size))[0] for _ in range(2*self.n) ]
        return [ (self.individuals[i], self. individuals[j]) for i, j in zip(positions[0::2], positions[1::2]) ]

    def evolution(self, generations, p_mutation, p_crossover, tournament_size, evolution_history = None):
        # Implements mutation and crossover algorithms over a certain number of generations.
        for generation in range(generations):
            
            # Crossover section of main genetic algorithm
            crossover_matches = self.tournament_crossover_selection(tournament_size)
            self.individuals = [ np.random.choice([son, ind1], size = 1, p = [p_crossover, 1 - p_crossover])[0] for ind1, ind2 in crossover_matches for son in self.cx_crossover(ind1, ind2) ]

            # Mutation section of main genetic algorithm
            mutations =  self.flat_mutation_selection(p_mutation)
            for ind, mut in zip(self.individuals, mutations): ind.mutation(mut)

            # Tracking generations during simulation 
            if generation % 10 == 0:
                print('- Generation {0} / {1} :'.format(generation, generations))


            # Recording of evolution informatin by generation 
            fitness = self.fitness()
            if evolution_history:
                evolution_history.record(self.individuals, fitness)

            # Call to to break out of main loop after convergence
            if fitness.count(0) >= self.n*0.95:
   
                break

# Class for tracking progress 
class History():

    def __init__(self, *args):
        self.attributes = args
        for attr in args: self.__dict__[attr] = []

    def record(self, *args):
        for attr, arg in zip(self.attributes, args):
            self.__dict__[attr] += [arg]

# ----------- Code to call for start of simulation --------------------------------#
myPop = Population(pop_size, dim)
evolution_history = History('individuals', 'fitness')
myPop.evolution(generations, p_mutation, p_crossover, tournament_size, evolution_history)
#----------------------------------------------------------------------------------#

# Function to convert values of individuals to matrix for representation
def ind_to_matrix(ind):
    size = len(ind)
    matrix = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            if ind[i] == j: matrix[i][j] = 1
    return matrix

# For plotting of data recorded in history

plt.ion()

individuals = [generation for generation in evolution_history.individuals]
mean_fitness = [np.mean(generation) for generation in evolution_history.fitness ]
fittest_individuals = [ ind[0].values for ind in individuals ]
fitness_gen = [min(gen) for gen in evolution_history.fitness]

plt.figure(figsize=(8,12))
plt.subplots_adjust(hspace = 0.5)
while True:
    i = 0
    for generation in evolution_history.fitness:

        counter = Counter( tuple(ind[i].values) for ind in individuals )
        most_common = counter.most_common(5)

        plt.suptitle('Generation {0}\n {1} individuals {2} x {2}'.format(i,pop_size, dim ))

        # Subplot for Histogram showing the fitness distribution of all individuals for each generation
        plt.subplot(2,2,1)
        plt.title('Fitness')
        plt.ylim(0, 1)
        plt.xlim( -0.5, max(evolution_history.fitness[0]) + 0.5)
        plt.hist(generation, density = True, bins = [i - 0.5 for i in range(max(evolution_history.fitness[0]) + 1)])

        # Subplot for showing average fitness evolution in comparison to fitness of fittest individual
        plt.subplot(2,2,2)
        plt.title('Average Fitness vs Generation')
        plt.plot(mean_fitness, label= 'Average fitness')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.plot(fitness_gen,label='Fittest Individual') 
        plt.axvline(i, color = 'r')
        plt.legend()

        # Subplot for showing solutions of fittest individual in generation
        plt.subplot(2,2,3)
        plt.title('Fittest individual')
        plt.xticks(np.arange( len(fittest_individuals[i]+1) ) - 0.5)
        plt.yticks(np.arange( len(fittest_individuals[i]+1) ) - 0.5)
        plt.grid()
        plt.tick_params(labelbottom = False, labelleft = False, bottom = False, left = False)
        plt.imshow(ind_to_matrix(fittest_individuals[i]), cmap = 'Greys', aspect ='auto')

        # Pie plot showing frequency of most common individuals
        unique_results = [ list(s) for s in set( tuple(ind.values) for ind in individuals[i] ) ]
        plt.subplot(2,2,4)
        plt.title('Frequency of the top individuals')
        plt.pie([mc[1] for mc in most_common], colors = ['steelblue', 'darkseagreen', 'goldenrod', 'coral', 'crimson'])

        plt.pause(0.3)
        plt.clf()

        i += 1
