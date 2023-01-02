import random

import function as fun
import numpy as np
from copy import copy


def genetic_algorithm(distance, flow, factory_list, population_size, selection_size, number_of_generation, selection_type='ranking', crossover_probability=1, mutation_probability=1, pmx_probability=1, cx_probability=1, ox_probability=1, swap_probability=1, inversion_probability=1, scramble_probability=1, min_value=-np.inf):
    current_generation = 0
    fitness_table = []
    min_values_list = []
    population = fun.generate_population(factory_list, population_size)
    for i in range(len(population)):
        fitness_table.append(fun.operative_function(
            population[i], distance, flow))
    current_min_value = min(fitness_table)
    best_individual_idx = fitness_table.index(current_min_value)
    best_individual = population[best_individual_idx]
#    print(best_individual, current_min_value)
    min_values_list.append(current_min_value)
    if current_min_value <= min_value:
        return best_individual, current_min_value, min_values_list
    while current_generation < number_of_generation:
        fitness_table_for_current_population = []
        selected_population = population
        if selection_type == 'roulette':
            selected_population = fun.selection(
                population, distance, flow, selection_size)
        elif selection_type == 'ranking':
            selected_population = fun.ranking_selection(
                population, distance, flow, selection_size)
        while len(selected_population) != len(population):
            genetic_operation = random.choices(list(fun.Operations), weights=[
                                               mutation_probability, crossover_probability])
            genetic_operation = genetic_operation[0]
            if genetic_operation == fun.Operations.crossover:
                par1_idx = np.random.randint(0, len(selected_population)-1)
                par2_idx = np.random.randint(0, len(selected_population)-1)
                cross_type = random.choices(list(fun.Crossover),
                                            weights=[cx_probability, pmx_probability, ox_probability])
                cross_type = cross_type[0]
                children_1 = selected_population[par1_idx]
                children_2 = selected_population[par2_idx]
                if cross_type == fun.Crossover.pmx:
                    children_1, children_2 = fun.pmx(
                        selected_population[par1_idx], selected_population[par2_idx])

                elif cross_type == fun.Crossover.cx:
                    children_1, children_2 = fun.cx(
                        selected_population[par1_idx], selected_population[par2_idx])

                elif cross_type == fun.Crossover.ox:
                    children_1, children_2 = fun.ox(
                        selected_population[par1_idx], selected_population[par2_idx])

                selected_population.append(children_1)
                if len(selected_population) != len(population):
                    selected_population.append(children_2)
            elif genetic_operation == fun.Operations.mutation:
                mutate_idx = np.random.randint(0, len(selected_population)-1)
                mut_type = random.choices(list(fun.Mutations),
                                          weights=[swap_probability, scramble_probability, inversion_probability])
                mut_type = mut_type[0]
                if mut_type == fun.Mutations.swap:
                    children = fun.swap_mutation(
                        selected_population[mutate_idx])
                    selected_population[mutate_idx] = children
                elif mut_type == fun.Mutations.inversion:
                    children = fun.inversion_mutation(
                        selected_population[mutate_idx])
                    selected_population[mutate_idx] = children
                elif mut_type == fun.Mutations.scramble:
                    children = fun.scramble_mutation(
                        selected_population[mutate_idx])
                    selected_population[mutate_idx] = children
            else:
                pass

        for i in range(len(selected_population)):
            fitness_table_for_current_population.append(
                fun.operative_function(selected_population[i], distance, flow))
        max_selected = min(fitness_table_for_current_population)
        if max_selected < current_min_value:
            current_min_value = max_selected
            idx_selected = fitness_table_for_current_population.index(
                current_min_value)
            best_individual = selected_population[idx_selected]
            if current_min_value <= min_value:
                break
        population = selected_population
        current_generation += 1
        min_values_list.append(current_min_value)
    return best_individual, current_min_value, min_values_list


def main():
    dist = [[np.inf, 5, 12, 11, 5, 9],
            [5, np.inf, 7, 5, 4, 7],
            [12, 7, np.inf, 1, 6, 10],
            [11, 5, 1, np.inf, 2, 4],
            [5, 4, 6, 2, np.inf, 4],
            [9, 7, 10, 6, 4, np.inf]]

    flow = [[np.inf, 4, 2, 2, 3, 1],
            [4, np.inf, 3, 5, 5, 8],
            [2, 3, np.inf, 9, 6, 4],
            [2, 5, 9, np.inf, 7, 9],
            [3, 5, 6, 7, np.inf, 2],
            [1, 8, 4, 9, 2, np.inf]]

    dist_matrix = np.array(dist)
    flow_matrix = np.array(flow)
    parcels_number = 6
    factory_number = 6
    fac_list = fun.create_fabric_list(parcels_number, factory_number)
    # fac_list = [2, 3, 4, 5, 0, 1]
    solution, value, list = genetic_algorithm(
        dist_matrix, flow_matrix, fac_list, 50, 30, 4, crossover_probability=1, mutation_probability=1, pmx_probability=1, cx_probability=1, ox_probability=1,
        swap_probability=1, inversion_probability=1, scramble_probability=1)
    # print(fac_list, fun.operative_function(
    #     fac_list, dist_matrix, flow_matrix), '   - wartosc poczatkowa')
    # print(solution, value, '   - wartosc koncowa')

    # dist, flow = fun.create_random_data_matrix(parcels_number, factory_number)
    # fac_list = fun.create_fabric_list(parcels_number, factory_number)
    # # print(fac_list)
    # # print(dist)
    # # print(flow)
    # solution, value = genetic_algorithm(
    #     dist, flow, fac_list, 30, 15, 20, 90, 10, crossover_type='pmx')
    print(solution)
    print(value)


if __name__ == "__main__":
    main()
