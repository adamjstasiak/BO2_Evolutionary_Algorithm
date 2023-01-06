from enum import Enum
import numpy as np
from random import sample, choices
from copy import copy


def create_random_data_matrix(number_parcels: int, number_of_factory: int):  # DONE
    """
    Function creating matrix with distances between parcels and flows between factories.
    Number_parcels = Amount of our available parcels
    Number_of_factory = Amount of fabric we need to locate

    Return:
        Distances - matrix with distance between parcels
        Flows - matrix with flows between each factory
`   """
    if number_of_factory > number_parcels:
        raise ValueError(
            "Number of factories is larger than number of parcels")
    else:
        distances = np.random.randint(
            1, 10+1, size=(number_parcels, number_parcels))
        distances = (distances+distances.T)/2
        np.fill_diagonal(distances, np.inf)
        flows = np.random.randint(
            1, 10+1, size=(number_of_factory, number_of_factory))
        flows = (flows+flows.T)/2
        np.fill_diagonal(flows, np.inf)
    return distances, flows


def operative_function(solution: list, distance_matrix: np.array, flow_matrix: np.array):  # DONE
    """
    Function calculating value of operative function for current solution.
    solution (List):
                    index - number of parcel;
                    element - number of factory.
    Return:
        Total value of operative function
    """
    if len(solution) > len(distance_matrix):
        raise ValueError("Length of solution is bigger than number of parcels")
    else:
        sum = 0
        for idx_i, el_i in enumerate(solution):
            for idx_j, el_j in enumerate(solution):
                if el_i != np.inf and el_j != np.inf:
                    if idx_i == idx_j or flow_matrix[el_i, el_j] == np.inf:
                        pass
                    else:
                        sum += distance_matrix[idx_i,
                            idx_j] * flow_matrix[el_i, el_j]

    return sum/2  # every distance and flow is added two times, so return divided by 2


def create_fabric_list(factory_number):  # DONE
    """
    Function creating list with number of each factories
    parcels_number = Amount of our available parcels
    Factory_number = Amount of factories we need to locate
    Return:
        Basic list which assigns first factory to parcel etc.
    """
    #factory_list = [np.inf for i in range(parcels_number)]

    factory_list = []
    for i in range(factory_number):
        factory_list.append(i)
    return factory_list


def create_random_solutions(fabric_list):
    """
    Function creating random beginning solutions for problem
    """
    return sample(fabric_list, len(fabric_list))


def generate_population(fabric_list: list, size_of_populations: int):
    """
    Function generating population of different solutions
    fabric_list - basic list with assigned first factory to first parcel etc.
    size_of_population - number of total solutions
    Return:
        List with starting population
    """
    population = []
    for i in range(size_of_populations):
        solution = sample(fabric_list, len(fabric_list))
        population.append(solution)
    return population


def selection(population, distance, flow, selection_size):  # DONE
    """
    Roulette wheel selection function return list with selected individuals 
    based on weighted random choice. 
    Population - list with our individuals to select
    Distances - matrix with distance between parcels
    Flow - matrix with flows between each factory
    Selection size - number od individuals we want to select.
    Return:
        List with selected individuals
    """
    selected_population = []
    fitness_table = []
    sum_p = 0
    selection_size = int(len(population)*selection_size/100)
    for i in range(len(population)):
        sum_p += operative_function(population[i], distance, flow)
    for i in range(len(population)):
        fitness_table.append(operative_function(
            population[i], distance, flow)/sum_p)
    for i in range(len(fitness_table)):
        fitness_table[i] = 1 - fitness_table[i]

    # for i in range(len(fitness_table)-1):
    #     fitness_table[i+1] += fitness_table[i]
    # print(fitness_table)

    copy_population = copy(population)
    for i in range(selection_size):
        selected_individual = choices(
            copy_population, weights=fitness_table)
        selected_population.append(selected_individual[0])
        idx_selected = copy_population.index(selected_individual[0])
        copy_population.remove(selected_individual[0])
        fitness_table.pop(idx_selected)
    return selected_population


def ranking_selection(population, distance, flow, selection_size):
    """Ranking type selection
    Population - list with our individuals to select
    Distances - matrix with distance between parcels
    Flow - matrix with flows between each factory
    Selection size - number od individuals we want to select.
    Return:
        List with selected individuals"""
    selected_population = []
    selection_size = int(len(population) * (selection_size/100))
    fitness_list = [(operative_function(individual,distance,flow),individual) for individual in population]
    fitness_list.sort()
    i = 0
    while i != selection_size:
        selected_population.append(fitness_list[i][1])
        i+=1
    return selected_population



def swap_mutation(solution):
    """
    Mutation function , swapping to random gens in individual
    """
    gen1 = np.random.randint(0, len(solution) - 1)
    gen2 = np.random.randint(0, len(solution) - 1)
    if gen1 != gen2:
        solution[gen1], solution[gen2] = solution[gen2], solution[gen1]
    else:
        pass
    return solution


def inversion_mutation(solution):
    """
    Inversion mutation function ,inverting gens in random section
    """
    gens = sample(range(0, len(solution)), 2)
    if gens[0] > gens[1]:
        gens[0], gens[1] = gens[1], gens[0]
    solution[gens[1]:gens[0]+1] = solution[gens[1]:gens[0]+1][::-1]
    return solution


def scramble_mutation(solution):
    """
    Scramble mutation function ,randomly swapping gens in random section
    """
    gens = sample(range(0, len(solution)), 2)
    if gens[0] > gens[1]:
        gens[0], gens[1] = gens[1], gens[0]
    scramble_tab = solution[gens[0]:gens[1]+1]
    new_queue = sample(scramble_tab, len(scramble_tab))
    solution[gens[0]:gens[1]+1] = new_queue
    return solution


class Mutations(Enum):
    swap = 1
    inversion = 2
    scramble = 3


class Operations(Enum):
    mutation = 1
    crossover = 2


class Crossover(Enum):
    pmx = 1
    cx = 2
    ox = 3


def pmx(parent_1: list, parent_2: list):
    """
    Function making Partially Matched Crossover between to parents creating two children.
    Crossover point is choose randomly.
    """
    not_erased = erase_inf(parent_1)
    erase_inf(parent_2)
    size = len(parent_1)
    k1 = np.random.randint(0, size - 1)
    k2 = np.random.randint(0, size - 1)
    if k2 < k1:
        temp = k1
        k1 = k2
        k2 = temp
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    children = [child_1, child_2]
    transposition = []
    for i in range(k1, k2 + 1):
        value_1 = parent_1[i]
        value_2 = parent_2[i]
        child_1[i] = value_2
        child_2[i] = value_1
        t = [value_1, value_2]
        trans = t.copy()
        to_remove = []
        for v in range(len(t)):
            for j in range(len(transposition)):
                if t[v] in transposition[j]:
                    if transposition[j].index(t[v]) == 0:
                        trans[v] = transposition[j][1]
                    else:
                        trans[v] = transposition[j][0]
                    to_remove.append(transposition[j])
        transposition.append(trans)
        for el in to_remove:
            transposition.remove(el)
    for child in children:
        for i in [i for i in range(0, k1)] + [i for i in range(k2+1, size)]:
            for j in range(len(transposition)):
                if child[i] in transposition[j]:
                    t = transposition[j]
                    if t.index(child[i]) == 0:
                        child[i] = t[1]
                    else:
                        child[i] = t[0]
    return_inf(parent_1, not_erased)
    return_inf(parent_2, not_erased)
    return_inf(child_1, not_erased)
    return_inf(child_2, not_erased)
    return child_1, child_2


def erase_inf(organism: list):
    fabric_counter = 0
    for el in organism:
        if el != np.inf:
            fabric_counter += 1
    not_erased = fabric_counter
    for i in range(len(organism)):
        if organism[i] == np.inf:
            fabric_counter += 1
            organism[i] = fabric_counter
    return not_erased


def return_inf(organism: list, not_erased: int):
    for i in range(len(organism)):
        if organism[i] > not_erased:
            organism[i] = np.inf


def cx(parent_1: list, parent_2: list):
    """
        Function making cycle crossover between to parents creating two children.
        """
    not_erased = erase_inf(parent_1)
    erase_inf(parent_2)
    k = np.random.randint(0, len(parent_1) - 1)
    start = k
    I = [k]
    while True:
        p1 = parent_1[k]
        p2 = parent_2[k]
        k = parent_1.index(p2)
        if k == start:
            break
        I.append(k)
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    for i in range(len(parent_1)):
        if i not in I:
            child_1[i] = parent_2[i]
            child_2[i] = parent_1[i]
    return_inf(parent_1, not_erased)
    return_inf(parent_2, not_erased)
    return_inf(child_1, not_erased)
    return_inf(child_2, not_erased)
    return child_1, child_2


def ox(parent_1: list, parent_2: list):
    """
    Function making Order Crossover between to parents creating two children.
    Crossover point is choose randomly.
    """
    not_erased = erase_inf(parent_1)
    erase_inf(parent_2)
    size = len(parent_1)
    k1 = np.random.randint(0, size - 1)
    k2 = np.random.randint(0, size - 1)
    if k2 < k1:
        temp = k1
        k1 = k2
        k2 = temp
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    supplement_list_1 = []
    supplement_list_2 = []
    for i in range(size):
        idx = i + k2 + 1
        if idx > size - 1:
            idx = idx - size
        supplement_list_1.append(parent_1[idx])
        supplement_list_2.append(parent_2[idx])
    for i in range(k1, k2 + 1):
        value_1 = parent_1[i]
        value_2 = parent_2[i]
        child_1[i] = value_2
        child_2[i] = value_1
        supplement_list_1.remove(value_2)
        supplement_list_2.remove(value_1)
    for i in range(size - (k2-k1+1)):
        idx = i + k2 + 1
        if idx > size-1:
            idx = idx - size
        child_1[idx] = supplement_list_1[i]
        child_2[idx] = supplement_list_2[i]

    return_inf(parent_1, not_erased)
    return_inf(parent_2, not_erased)
    return_inf(child_1, not_erased)
    return_inf(child_2, not_erased)
    return child_1, child_2


