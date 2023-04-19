import random

import numpy as np
from deap import base
from deap import creator
from deap import gp
from deap import tools
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from custom_algorithms import eaSimple
from selection import selectMapElites


# Define new functions
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# Load the diabetes dataset
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

# Standardize the input features and target values based on the training data
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

pset = gp.PrimitiveSet("MAIN", X_train.shape[1])
pset.addPrimitive(np.add, 2, name="vadd")
pset.addPrimitive(np.subtract, 2, name="vsub")
pset.addPrimitive(np.multiply, 2, name="vmul")
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(np.negative, 1, name="vneg")
pset.addPrimitive(np.cos, 1, name="vcos")
pset.addPrimitive(np.sin, 1, name="vsin")
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
for i in range(X_train.shape[1]):
    pset.renameArguments(**{f"ARG{i}": f"x{i}"})

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the sum of squared difference between the expression
    predicted_values = func(*X_train.T)
    individual.predicted_values = predicted_values
    diff = np.mean((predicted_values - y_train) ** 2)
    return diff,


toolbox.register("evaluate", evalSymbReg)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(0)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    eaSimple(pop, toolbox, 0.9, 0.1, 40, stats, halloffame=hof)

    # Perform ensemble prediction on test data using individuals in pop
    func_list_pop = [toolbox.compile(expr=ind) for ind in pop]
    y_pred_list_pop = [func(*X_test.T) for func in func_list_pop]
    y_pred_ensemble_pop = np.mean(y_pred_list_pop, axis=0)
    mse_test_pop = np.mean((y_pred_ensemble_pop - y_test) ** 2)
    print("Mean squared error on test data for pop ensemble:", mse_test_pop)

    # Perform ensemble prediction on test data using individuals in hof
    func_list_hof = [toolbox.compile(expr=ind) for ind in hof]
    y_pred_list_hof = [func(*X_test.T) for func in func_list_hof]
    y_pred_ensemble_hof = np.mean(y_pred_list_hof, axis=0)
    mse_test_hof = np.mean((y_pred_ensemble_hof - y_test) ** 2)
    print("Mean squared error on test data for hof ensemble:", mse_test_hof)
    return pop, stats, hof


def no_selection(x, k):
    return x[-k:]


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    # Mean squared error on test data for pop ensemble: 0.6523436598913606
    # Mean squared error on test data for hof ensemble: 0.5808271583069419
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("environmental_selection", no_selection)
    main()

    # Mean squared error on test data for pop ensemble: 0.6422934035567954
    # Mean squared error on test data for hof ensemble: 0.5980863330099079
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("environmental_selection", no_selection)
    main()

    # Mean squared error on test data for pop ensemble: 0.5462900755024485
    # Mean squared error on test data for hof ensemble: 0.5525759058997444
    toolbox.register("select", tools.selRandom)
    toolbox.register("environmental_selection", selectMapElites, target=y_train)
    main()
