from deap import creator, base, tools, algorithms
from sympy.core import function

from TestApproximateFunctions import test_spline
from cec2017.functions import f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
import numpy as np
from matplotlib import pyplot as plt
from Utils import init_method


def create_run_ea(f: function, dimensions, population_size, num_generations):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", init_method)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dimensions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        return f(individual),

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    training_set = []
    for i in population:
        training_set.append(i)
    NGEN=num_generations
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        for o in offspring:
            training_set.append(o)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, k=1)
    return training_set, best_ind
