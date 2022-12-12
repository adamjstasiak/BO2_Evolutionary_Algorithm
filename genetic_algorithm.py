import function as fun
import numpy as np
from copy import copy

def genetic_algorith(distance,flow,factory_list,population_size,selection_size,number_of_generation,crossover_probability,mutation_probability,min_value = -np.inf,croosover_type = 'pmx'):
    current_generation  = 0
    fitness_table = []
    population = fun.generate_population(factory_list,population_size)
    for i in range(len(population)):
            fitness_table.append(fun.operative_function(population[i],distance,flow))
    max_value = max(fitness_table)
    best_individual_idx = fitness_table.index(max_value)
    best_individual = population[best_individual_idx]
    print(best_individual,max_value)
    if max_value <= min_value:
        return best_individual , max_value
    while current_generation <= number_of_generation:
        fitness_table_for_current_population = []
        selected_population = fun.selection(population,distance,flow,selection_size)
        cross_p = np.random.randint(0,100+1)
        if cross_p <= crossover_probability:
            par1_idx = np.random.randint(0,len(selected_population)-1)
            par2_idx = np.random.randint(0,len(selected_population)-1)
            if croosover_type == 'pmx':
                children_1,children_2 = fun.pmx(selected_population[par1_idx],selected_population[par2_idx])
                selected_population[par1_idx] = children_1
                selected_population[par2_idx] = children_2
            elif croosover_type == 'cx':
                children_1,children_2 = fun.cx(selected_population[par1_idx],selected_population[par2_idx])
                selected_population[par1_idx] = children_1
                selected_population[par2_idx] = children_2
        else:
            pass
        mut_p = np.random.randint(0,100+1)
        if mut_p <= mutation_probability:
            mutate_idx = np.random.randint(0,len(selected_population)-1)
            children = fun.mutation(selected_population[mutate_idx])
            selected_population[mutate_idx] = children
        else:
            pass
        for i in range(len(selected_population)):
            fitness_table_for_current_population.append(fun.operative_function(selected_population[i],distance,flow))
        max_selected = max(fitness_table_for_current_population)
        if max_selected < max_value:
            max_value = max_selected
            idx_selcted = fitness_table_for_current_population.index(max_value)
            best_individual = selected_population[idx_selcted]
            if max_value <= min_value:
                break
        population = selected_population
        current_generation += 1
    
    return best_individual , max_value
    
def main():
    parcels_number = 15
    factory_number = 12
    dist , flow  = fun.create_random_data_matrix(parcels_number,factory_number)
    fac_list = fun.create_fabric_list(parcels_number,factory_number)
    # print(fac_list)
    # print(dist)
    # print(flow)
    solution, value = genetic_algorith(dist,flow,fac_list,30,15,20,90,10,croosover_type='pmx') 
    print(solution)
    print(value)
   
if __name__ == "__main__":
    main()


