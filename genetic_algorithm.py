import random
import function as fun
import numpy as np

def genetic_algorithm(distance, flow, factory_list, population_size, selection_size, number_of_generation,populatian_type='mi+lambda', selection_type='ranking', crossover_probability=1, mutation_probability=1, pmx_probability=1, cx_probability=1, ox_probability=1, swap_probability=1, inversion_probability=1, scramble_probability=1, stop_count=80):
    current_generation = 0
    fitness_table = []
    min_values_list = []
    operand_type = []
    crossover_type = []
    mutation_type = []
    crossover_value = []
    mutation_value = []
    counter = 0
    population = fun.generate_population(factory_list, population_size)
    for i in range(len(population)):
        fitness_table.append(fun.operative_function(
            population[i], distance, flow))
    current_min_value = min(fitness_table)
    best_individual_idx = fitness_table.index(current_min_value)
    best_individual = population[best_individual_idx]
    min_values_list.append(current_min_value)
    while current_generation < number_of_generation:
        new_population = []
        fitness_table_for_current_population = []
        if selection_type == 'roulette':
            selected_population = fun.selection(
                population, distance, flow, selection_size)
        elif selection_type == 'ranking':
            selected_population = fun.ranking_selection(
                population, distance, flow, selection_size)
        while True:
            genetic_operation = random.choices(list(fun.Operations), weights=[
                                               mutation_probability, crossover_probability])
            genetic_operation = genetic_operation[0]
            if genetic_operation == fun.Operations.crossover:
                par1_idx = np.random.randint(0, len(selected_population)-1)
                par2_idx = np.random.randint(0, len(selected_population)-1)
                cross_type = random.choices(list(fun.Crossover),
                                            weights=[cx_probability, pmx_probability, ox_probability])
                operand_type.append('Crossover')
                cross_type = cross_type[0]
                children_1 = selected_population[par1_idx]
                children_2 = selected_population[par2_idx]
                if cross_type == fun.Crossover.pmx:
                    children_1, children_2 = fun.pmx(
                        selected_population[par1_idx], selected_population[par2_idx])
                    crossover_type.append('PMX')
                    mutation_type.append(np.NaN)
                    if fun.operative_function(children_2,distance,flow) > fun.operative_function(children_1,distance,flow):
                        bufor = children_1
                        children_1 = children_2
                        children_2 = bufor 
                    crossover_value.append(('PMX',fun.operative_function(children_1,distance,flow),current_min_value-fun.operative_function(children_1,distance,flow)))
                elif cross_type == fun.Crossover.cx:
                    children_1, children_2 = fun.cx(
                        selected_population[par1_idx], selected_population[par2_idx])
                    crossover_type.append('CX')
                    mutation_type.append(np.NaN)
                    if fun.operative_function(children_2,distance,flow) > fun.operative_function(children_1,distance,flow):
                        bufor = children_1
                        children_1 = children_2
                        children_2 = bufor 
                    crossover_value.append(('CX',fun.operative_function(children_1,distance,flow),current_min_value-fun.operative_function(children_1,distance,flow)))
                elif cross_type == fun.Crossover.ox:
                    children_1, children_2 = fun.ox(
                        selected_population[par1_idx], selected_population[par2_idx])
                    crossover_type.append('OX')
                    mutation_type.append(np.NaN)
                    if fun.operative_function(children_2,distance,flow) > fun.operative_function(children_1,distance,flow):
                        bufor = children_1
                        children_1 = children_2
                        children_2 = bufor 
                    crossover_value.append(('OX',fun.operative_function(children_1,distance,flow),current_min_value-fun.operative_function(children_1,distance,flow)))
                new_population.append(children_1)
                if len(new_population) != len(population):
                    new_population.append(children_2)
            elif genetic_operation == fun.Operations.mutation:
                operand_type.append('mutation')
                mutate_idx = np.random.randint(0, len(selected_population)-1)
                mut_type = random.choices(list(fun.Mutations),
                                          weights=[swap_probability, scramble_probability, inversion_probability])
                mut_type = mut_type[0]
                if mut_type == fun.Mutations.swap:
                    children = fun.swap_mutation(
                        selected_population[mutate_idx])
                    mutation_type.append('Swap')
                    crossover_type.append(np.NaN)
                    mutation_value.append(('Swap',fun.operative_function(children,distance,flow),current_min_value-fun.operative_function(children,distance,flow)))
                elif mut_type == fun.Mutations.inversion:
                    children = fun.inversion_mutation(
                        selected_population[mutate_idx])
                    mutation_type.append('Inversion')
                    crossover_type.append(np.NaN)
                    mutation_value.append(('Inversion',fun.operative_function(children,distance,flow),current_min_value-fun.operative_function(children,distance,flow)))
                elif mut_type == fun.Mutations.scramble:
                    children = fun.scramble_mutation(
                        selected_population[mutate_idx])
                    mutation_type.append('Scramble')
                    crossover_type.append(np.NaN)
                    mutation_value.append(('Scramble',fun.operative_function(children,distance,flow),current_min_value-fun.operative_function(children,distance,flow)))
                if populatian_type == 'mi':
                    new_population.append(children)
                if populatian_type == 'mi+lambda':
                    selected_population[mutate_idx] = children
            else:
                pass
            if populatian_type == 'mi+lambda':
                if len(new_population) == (len(population)-len(selected_population)):
                    new_population = new_population + selected_population
            if len(new_population) == len(population):
                break
        for i in range(len(selected_population)):
            fitness_table_for_current_population.append(
                fun.operative_function(selected_population[i], distance, flow))
        max_selected = min(fitness_table_for_current_population)
        if max_selected >= current_min_value:
            counter += 1
        if max_selected < current_min_value:
            counter = 0
            current_min_value = max_selected
            idx_selected = fitness_table_for_current_population.index(
                current_min_value)
            best_individual = selected_population[idx_selected]
        if counter == stop_count:
            break
        population = new_population
        current_generation += 1
        min_values_list.append(current_min_value)
    return best_individual, current_min_value, min_values_list,operand_type,crossover_type,mutation_type,crossover_value,mutation_value