import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

 #TODO: Visualize the steps of Differential Evolution
 #TODO: Implement variants of the algorithm (have separate mutation and recombination methods in the class)
 #TODO: Try to use cython to speed this up?


class DifferentialEvolution:

    def __init__(self, f, limits, seed=None):
        """
            Expects limits as list of intervals
        """
        self.f = f
        self.dimensions = len(limits)
        self.limits = np.asarray(limits).T
        self.seed = seed
        self.fitness_history = None


    def _rescale(self, arr):
        lower, upper = self.limits
        diff = np.abs(upper - lower)
        return lower + arr * diff

    def _fitness(self, arr):
        return np.fromiter((self.f(ind) for ind in arr), float)

    @staticmethod
    def _current_to_best_mutation(mut, current_idx, best_idx, population):
        popsize = len(population)
        idxs = [idx for idx in range(popsize) if idx != best_idx and idx != current_idx]

        x1, x2 = population[np.random.choice(idxs, size=2, replace=False)]

        current = population[current_idx]
        best = population[best_idx]

        mutant = current + mut * (best - current) + mut * (x1 - x2)
        return np.clip(mutant, 0, 1)

    @staticmethod
    def _rand_mutation(mut, current_idx, population):
        popsize = len(population)
        idxs = [idx for idx in range(popsize) if idx != current_idx]

        x1, x2, x3 = np.random.choice(idxs, size=3, replace=False)

        mutant = x1 + mut * (x2 - x3)

        return np.clip(mutant, 0, 1)



    def optimize(self, popsize=20, iterations=100, mut=0.5, crossp=0.7):
        np.random.seed(self.seed)
        dims = self.dimensions

        fitness_history = np.zeros(iterations)

        # Generate and rescale initial population
        pop = np.random.rand(popsize, dims)
        pop_resc = self._rescale(pop)

        # Compute initial fitness function
        fitness = self._fitness(pop_resc)

        # Compute the population member that minimizes the initial fitness
        best_idx = np.argmin(fitness)
        best = pop_resc[best_idx]

        for i in range(iterations):
            for j in range(popsize):
                # Randomly select 3 vectors from the population
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Calculate the mutation vector
                mutant = np.clip(a + mut * (b-c), 0, 1)

                # Determine the components where recombination occurs
                cross_points = np.random.rand(dims) < crossp

                if not np.any(cross_points):
                    cross_points[np.random.randint(0,dims)] = True

                trial = np.where(cross_points, mutant, pop[j])
                trial_resc = self._rescale(trial)

                score = self.f(trial_resc)

                if score < fitness[j]:
                    fitness[j] = score
                    pop[j] = trial
                    if score < fitness[best_idx]:
                        best_idx = j
                        best = trial_resc

            fitness_history[i] = fitness[best_idx]

        self.fitness_history = fitness_history


        return best, fitness[best_idx]

class Testing:

    def __init__(self, limits=None, **kwargs):
        self.limits = limits
        self.params = kwargs

    @staticmethod
    def plot_fitness(fitness_history):
        x = [i for i in range(len(fitness_history))]
        y = fitness_history
        plt.plot(x,y)
        plt.show()

    def plot_function(self, f):
        x0, x1 = self.limits[0]
        y0, y1 = self.limits[1]

        x = np.arange(x0, x1, 0.1)
        y = np.arange(y0, y1, 0.1)

        xs, ys = np.meshgrid(x,y)
        z = [f([u,v]) for u,v in zip(xs,ys)]

        plt.contourf(x,y,z)
        plt.show()

    def quadraticTest(self):
        print("Starting test for the function f(x,y) = x^2 + y^2")
        print("="*50)
        f = lambda x: sum(i ** 2 for i in x)
        problem = DifferentialEvolution(f, self.limits)
        pt, value = problem.optimize(**self.params)
        print(f"Minimum is at:\t\t\t {np.round(pt,decimals=6)}")
        print(f"Minimal value is:\t\t {np.round(value, decimals=6)}.")
        print("="*50)
        self.plot_function(f)
        self.plot_fitness(problem.fitness_history)


    def ackleyTest(self):
        def ackley_function(x):
            x1, x2 = x
            #returns the point value of the given coordinate
            part_1 = -0.2*np.sqrt(0.5*(x1*x1 + x2*x2))
            part_2 = 0.5*(np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))
            value = np.exp(1) + 20 -20*np.exp(part_1) - np.exp(part_2)
            #returning the value
            return value

        print("Starting test for the Ackley function.")
        print("="*50)
        problem = DifferentialEvolution(ackley_function, self.limits)
        pt, value = problem.optimize(**self.params)
        print(f"Minimum is at:\t\t {np.round(pt,decimals=6)}")
        print(f"Minimal value is:\t\t {np.round(value, decimals=6)}.")
        print("="*50)
        self.plot_function(ackley_function)
        self.plot_fitness(problem.fitness_history)







if __name__ == '__main__':
    limits = [(-5,5)]*2

    qtest = Testing(limits)
    qtest.quadraticTest()
    qtest.ackleyTest()










