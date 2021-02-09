import numpy as np
import matplotlib.pyplot as plt
from mutations import current_to_best_mutation, rand_mutation


# TODO: Visualize the steps of Differential Evolution
# TODO: Implement variants of the algorithm (have separate mutation and recombination methods in the class)


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
        self.generation_history = None
        self.best_in_generation = None

    def _rescale(self, arr):
        lower, upper = self.limits
        diff = np.abs(upper - lower)
        return lower + arr * diff

    def _fitness(self, arr):
        return np.fromiter((self.f(ind) for ind in arr), float)

    @staticmethod
    @profile
    def _current_to_best_mutation(mut, current_idx, best_idx, population, popsize):

        idx1, idx2 = np.random.choice(popsize - 2, size=2, replace=False)

        lo, hi = (current_idx, best_idx) if current_idx < best_idx else (best_idx, current_idx)

        # This is a bijection from range(popsize - 2) to range(popsize)\{current, best}
        idx1 = idx1 if idx1 < lo else idx1 + 2 if idx2 >= hi else idx1 + 1
        idx2 = idx2 if idx2 < lo else idx2 + 2 if idx2 >= hi else idx2 + 1

        x1 = population[idx1]
        x2 = population[idx2]
        current = population[current_idx]
        best = population[best_idx]

        return current_to_best_mutation(current, best, x1, x2, mut)

    # Replacing with above version reduces the running time by 50%!! The bottleneck is the np.random.choice function
    # def _current_to_best_mutation(mut, current_idx, best_idx, population):
    #     popsize = len(population)
    #     idxs = np.fromiter((idx for idx in range(popsize) if idx != best_idx and idx != current_idx), int)

    #     x1, x2 = population[np.random.choice(idxs, size=2, replace=False)]

    #     current = population[current_idx]
    #     best = population[best_idx]

    #     result = current_to_best_mutation(current, best, x1, x2, mut)
    #     return result

    @staticmethod
    def _rand_mutation(mut, current_idx, best_idx, population, popsize):
        idxs = np.fromiter((idx for idx in range(popsize) if idx != current_idx), int)

        x1, x2, x3 = population[np.random.choice(idxs, size=3, replace=False)]

        result = rand_mutation(x1, x2, x3, mut)
        return result

    def optimize(
        self,
        popsize=20,
        iterations=100,
        mut=0.5,
        crossp=0.7,
        strategy="curr_to_best",
        generation_history=False,
        track_best=False,
        initial_pop=None
    ):

        STRATEGIES = {
            "rand_mutation": self._rand_mutation,
            "curr_to_best": self._current_to_best_mutation,
        }

        mutation_function = STRATEGIES.get(strategy, "curr_to_best")

        np.random.seed(self.seed)
        dims = self.dimensions
        fitness_history = np.zeros(iterations)

        # Generate and rescale initial population

        if initial_pop is not None:
            pop = initial_pop
            popsize = len(pop)
        else:
            pop = np.random.rand(popsize, dims)

        pop_resc = self._rescale(pop)

        # Compute initial fitness function
        fitness = self._fitness(pop_resc)

        # Compute the population member that minimizes the initial fitness
        best_idx = np.argmin(fitness)
        best = pop_resc[best_idx]

        if generation_history:
            self.generation_history = [pop_resc]

        if track_best:
            self.best_in_generation = [best]

        for i in range(iterations):
            for j in range(popsize):
                # mutant = self._rand_mutation(mut, j, pop)
                mutant = mutation_function(mut, j, best_idx, pop, popsize)

                cross_points = np.random.rand(dims) < crossp

                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dims)] = True

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

            if track_best:
                self.best_in_generation.append(list(best))
            if generation_history:
                self.generation_history.append(list(self._rescale(pop)))

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
        plt.plot(x, y)
        plt.show()

    def plot_function(self, f):
        x0, x1 = self.limits[0]
        y0, y1 = self.limits[1]

        x = np.arange(x0, x1, 0.1)
        y = np.arange(y0, y1, 0.1)

        xs, ys = np.meshgrid(x, y)
        z = [f([u, v]) for u, v in zip(xs, ys)]

        plt.contourf(x, y, z)
        plt.show()

    def quadraticTest(self, plot=False):
        print("Starting test for the function f(x,y) = x^2 + y^2")
        print("=" * 50)
        def f(x):
            return np.sum(x ** 2)

        problem = DifferentialEvolution(f, self.limits)
        pt, value = problem.optimize(**self.params)
        print(f"Minimum is at:\t\t\t {np.round(pt,decimals=10)}")
        print(f"Minimal value is:\t\t {np.round(value, decimals=10)}.")
        print("=" * 50)
        if plot:
            self.plot_function(f)
            self.plot_fitness(problem.fitness_history)

    def ackleyTest(self, plot=False):
        def ackley_function(x):
            x1, x2 = x
            # returns the point value of the given coordinate
            part_1 = -0.2 * np.sqrt(0.5 * (x1 * x1 + x2 * x2))
            part_2 = 0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
            value = np.exp(1) + 20 - 20 * np.exp(part_1) - np.exp(part_2)
            # returning the value
            return value

        print("Starting test for the Ackley function.")
        print("=" * 50)
        problem = DifferentialEvolution(ackley_function, self.limits)
        pt, value = problem.optimize(**self.params)
        print(f"Minimum is at:\t\t\t {np.round(pt,decimals=10)}")
        print(f"Minimal value is:\t\t {np.round(value, decimals=10)}.")
        print("=" * 50)
        if plot:
            self.plot_function(ackley_function)
            self.plot_fitness(problem.fitness_history)


if __name__ == "__main__":
    #Benchmark settings
    limits = [(-50, 50)] * 25
    qtest = Testing(limits, popsize=100, iterations=5000)
    qtest.quadraticTest(plot=False)
    # qtest.ackleyTest(plot=False)
