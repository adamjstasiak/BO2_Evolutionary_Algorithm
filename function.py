import numpy as np
import itertools as it 
from random import sample,shuffle

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


def operative_function(sollution:list, distance_matrix:np.array, flow_matrix:np.array):
    """
    Function calculating value of operative function for current solution.
    sollution (List):
                    index - nuber of parcel; 
                    element - number of factory.
    Return:
        Total value of operative function
    """
    if len(sollution) > len(distance_matrix[0]):
        raise ValueError("Length of sollution is bigger than number of parcels")
    else:
        sum = 0
        for idx_i, el_i in enumerate(sollution):
            for idx_j, el_j in enumerate(sollution):
                if idx_i == idx_j or flow_matrix[el_i, el_j] == np.inf or (el_i or el_j == np.inf):
                    pass
                else:
                    sum += distance_matrix[idx_i, idx_j] + flow_matrix[el_i, el_j]

    return sum/2 #kazdy dystans i przeplyw jest uwzgledniony dwa razy wiec dziele na 2
    
    
def create_fabric_list(parcels_number:list,factory_number:list):
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
        Total statring population
    """
    population = []
    for i in range(size_of_populations):
        shuffle(fabric_list)
        population.append(fabric_list)
    return population


def mutation(solution):
    
    return None

    

def main():
    parcels_number = 8
    factory_number = 7
    dist , flow  = create_random_data_matrix(parcels_number,factory_number)
    fac_list = create_fabric_list(parcels_number,factory_number)
    print(fac_list)
    print(dist)
    print(flow)
    print(operative_function(fac_list, dist, flow)) 
    print(generate_population(fac_list,20))

if __name__ == "__main__":
    main()
