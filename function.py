import numpy as np
import itertools as it 
from random import sample,shuffle,choices

def create_random_data_matrix(number_parcels:int, number_of_factory:int):
    """
    Function creating matrix with distances between parcels and flows beetween factories.
    Number_parcels = Amount of our availaible parcels
    Number_of_factory = Amoount of fabric we need to locate
    
    Return:
        Distances - matrix with distance between parcels
        Flows - matrix with flows between each factory
`   """
    if number_of_factory > number_parcels:
        raise ValueError("Number of factories is larger than number of parcels")
    else:
        distances = np.random.randint(1, 10+1, size=(number_parcels,number_parcels))
        distances = (distances+distances.T)/2
        np.fill_diagonal(distances, np.inf)
        flows = np.random.randint(1, 10+1, size=(number_of_factory,number_of_factory))
        flows = (flows+flows.T)/2
        np.fill_diagonal(flows, np.inf)
    return distances, flows


def operative_function(solution: list, distance_matrix: np.array, flow_matrix: np.array):
    """
    Function calculating value of operative function for current solution.
    solution (List):
                    index - number of parcel;
                    element - number of factory.
    Return:
        Total value of operative function
    """
    if len(solution) > len(distance_matrix[0]):
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
                                               idx_j] + flow_matrix[el_i, el_j]

    return sum/2  # every distance and flow is added two times, so return divided by 2
    
def create_fabric_list(parcels_number,factory_number):
    """
    Function creating list with number of each factories
    parcels_number = Amount of our availaible parcels
    Factory_number = Amount of factories we need to locate
    Return:
        Basic list which assings first factory to parcel etc.
    """
    factory_list = [np.inf for i in range(parcels_number) ]
    for i in range(factory_number):
        factory_list[i] = i
    return factory_list


def generate_population(fabric_list:list, size_of_populations:int): 
    """
    Function generating population of different sollutions
    fabric_list - basic list withh assinged first factory to first parcel etc.
    size_of_population - number of total sollutions
    Return:
        List with starting population
    """
    population = []
    for i in range(size_of_populations):
        solution = sample(fabric_list,len(fabric_list))
        population.append(solution)
    return population


def selection(population,distance,flow,selection_size):
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
    fitness_table = []
    sum_p = 0
    for i in range(len(population)):
        sum_p += operative_function(population[i],distance,flow) 
    for i in range(len(population)):
        fitness_table.append(operative_function(population[i],distance,flow)/sum_p)
    for i in range(len(fitness_table)-1):
         fitness_table[i+1] += fitness_table[i] 
    for i in range(len(fitness_table)):
        fitness_table[i] = 1 - fitness_table[i] 
    selected_population = choices(population,cum_weights=fitness_table,k=selection_size)
    return selected_population



def mutation(solution):
    """
    Mutation fuction , swaping to random gens in individual

    """
    gen1 = np.random.randint(0,len(solution) - 1)
    gen2 = np.random.randint(0,len(solution) - 1)
    solution[gen1],solution[gen2] = solution[gen2],solution[gen2]
    return solution


def pmx(parent_1:list, parent_2:list): #TODO zrobić pmx
    """
    Function making Partially Matched Crossover between to parents creating two childrens.
    Crossover point is choose randomly.
    """
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
    return child_1, child_2


def cx(parent_1:list,parent_2:list):
    """
        Function making cycle crossover between to parents creating two childrens.
        """
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
    return child_1, child_2


def main():
    parcels_number = 10
    factory_number = 10
    dist , flow  = create_random_data_matrix(parcels_number,factory_number)
    fac_list = create_fabric_list(parcels_number,factory_number)
    population = generate_population(fac_list,100)
    # print(fac_list)
    # print(dist)
    # print(flow)
    print(population)
    print(operative_function(fac_list, dist, flow))
    p = generate_population(fac_list,2)
    print(p)
    c = pmx(p[0], p[1])
    print(c)
#    print(selection(population,dist,flow,20))
   
if __name__ == "__main__":
    main()
