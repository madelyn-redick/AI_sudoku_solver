"""
This file uses AI evolutionary algorithms sovle Sudoku puzzles
By Madelyn Redick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time



def get_structure(full_arr, structure, idx, sg_idx=0):
    """ returns a row, column, or subgrid at given index(es)

    Args:
        full_arr (numpy.ndarray): 9x9 puzzle
        structure (string): row, column, or subgrid
        idx (int): starting index of structure
        sg_idx (int): y index of subgrid, optional, default=0

    Returns:
        arr (numpy.ndarray): desired structure type at given index(es) of shape (9,)
    """
    # TODO DATA VALIDTION
    if structure == "row":
        arr = full_arr[idx]
    elif structure == "column":
        arr = full_arr[:, idx]
    elif structure == "subgrid":
        sub_x, sub_y = get_subgrid_coordinates(idx, sg_idx)
        arr = np.array(list(full_arr[sub_x][sub_y:sub_y+3]) + list(full_arr[sub_x+1][sub_y:sub_y+3]) + list(full_arr[sub_x+2][sub_y:sub_y+3]))

    return arr

def get_subgrid_coordinates(x, y):
    """ get coordinates of top left cell of the subgrid containing cell (x, y)

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_ TODO
    """
    if x <= 2:
        sub_x = 0
    elif x <= 5:
        sub_x = 3
    else:
        sub_x = 6

    if y <= 2:
        sub_y = 0
    elif y <= 5:
        sub_y = 3
    else:
        sub_y = 6

    return sub_x, sub_y

def get_replacement_indexes(full_arr):
    replace = []
    for i in range(9):
        for j in range(9):
            if full_arr[i][j] == 0:
                replace.append((i, j))
    return replace

def replace_structure(full_array, sub_array, structure, idx, sg_idx=0):
    """ replaces a partially solved row/column/subgrid with its original

    Args:
        full_array (numpy.ndarray): 9x9 puzzle
        sub_array (numpy.ndarray): row/column/subgrid of a puzzle
        structure (string): row, column, or subgrid
        idx (int): starting index of structure
        sg_idx (int): y index of subgrid, optional, default=0

    Returns:
        full_array: partially filled 9x9 puzzle 
    """
    if structure == "row":
        full_array[idx] = sub_array
    elif structure == "column":
        full_array[:, idx] = sub_array
    elif structure == "subgrid":
        full_array[idx][sg_idx:sg_idx+3] = sub_array[:3]
        full_array[idx+1][sg_idx:sg_idx+3] = sub_array[3:6]
        full_array[idx+2][sg_idx:sg_idx+3] = sub_array[6:]

    return full_array

def get_remaining_values(arr):
    """ finds values not already present in the row/column/subgrid

    Args:
        arr (numpy.ndarray): represents row/column/subgrid of a puzzle

    Returns:
        rem_vals (numpy.ndarray): values not already present in the row/column/subgrid
    """
    rem_vals = [x+1 for x in range(9)]
    for v in arr:
        if v != 0 and v in rem_vals:
            rem_vals.remove(v)

    return np.array(rem_vals)

def get_accuracy(unsolved, solved, generated_sol, percentage=True):
    # check how many generated values were correctly placed
    # TODO

    # get indexes of values to be filled
    to_fill = get_replacement_indexes(unsolved)
    
    totals = {"correct":0, "zero":0, "incorrect":0} 
    for loc in to_fill:
        if generated_sol[loc[0]][loc[1]] == solved[loc[0]][loc[1]]:
            totals["correct"] += 1
        elif generated_sol[loc[0]][loc[1]] == 0:
            totals["zero"] += 1
        else:
            totals["incorrect"] += 1

    if percentage:
        # of all values needed to be filled, how many were correct
        return totals["correct"]/len(to_fill)
    return totals

def compare_parameters(to_solve, solved, population_sizes, generations):
    #TODO
    results = {}
    for evolution in range(len(population_sizes)):
        pop_size = population_sizes[evolution]
        gen_size = generations[evolution]
        sol, fitness_score, time = genetic_algorithm(to_solve, solved, pop_size, gen_size)
        accuracy = get_accuracy(to_solve, solved, sol)
        results[evolution] = {"solution":sol, "fitness_score":fitness_score, "accuracy":accuracy, "pop_size":pop_size, "gen_size":gen_size, "seconds":time}

    return results

def plot_compare_parameters(results):
    #TODO
    # Extract everything with their keys
    evolution_ids = list(results.keys())
    fitness_scores = [results[k]["fitness_score"] for k in evolution_ids]
    accuracies = [results[k]["accuracy"] for k in evolution_ids]
    pop_sizes = [results[k]["pop_size"] for k in evolution_ids]
    gen_sizes = [results[k]["gen_size"] for k in evolution_ids]
    times = [results[k]["seconds"] for k in evolution_ids]

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter 1: Accuracy vs Fitness Score
    axs[0].scatter(fitness_scores, accuracies, c='lightblue', marker='o')
    for i, k in enumerate(evolution_ids):
        axs[0].annotate(str(k), (fitness_scores[i], accuracies[i]), fontsize=8, alpha=0.7)
    axs[0].set_xlabel("Fitness Score")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy vs Fitness Score")
    axs[0].grid(True)

    # Scatter 2: Population Size vs Generation Size
    axs[1].scatter(pop_sizes, gen_sizes, c='orange', marker='s')
    for i, k in enumerate(evolution_ids):
        axs[1].annotate(str(k), (pop_sizes[i], gen_sizes[i], times[i]), fontsize=8, alpha=0.7)
    axs[1].set_xlabel("Population Size")
    axs[1].set_ylabel("Generation Size")
    axs[1].set_title("Population vs Generation")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("acc_fit_vs_pop_gen.png")

def compare_puzzles(indexes, all_unsolved, all_solved, population_size, generations, hills):
    results = {}
    pop_gen_idx = 0
    for idx in indexes:
        to_solve = all_unsolved[idx]
        solved = all_solved[idx]
        sol, fitness_score, time = genetic_algorithm(to_solve, solved, population_size[pop_gen_idx], generations[pop_gen_idx], hills[pop_gen_idx])
        accuracy = get_accuracy(to_solve, solved, sol)
        results[idx] = {"solution":sol, "fitness_score":fitness_score, "accuracy":accuracy, "pop_size":population_size[pop_gen_idx], "gen_size":generations[pop_gen_idx], "seconds":time}
        pop_gen_idx += 1
    return results

def draw_grid(ax, grid, title):
            ax.set_title(title)
            ax.set_xlim(0, 9)
            ax.set_ylim(0, 9)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(0, 10, 1))
            ax.set_yticks(np.arange(0, 10, 1))
            ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.set_aspect('equal')

            # Draw thin grid
            for x in range(10):
                lw = 2 if x in [0, 3, 6, 9] else 1
                ax.axvline(x, color='black', linewidth=lw)
                ax.axhline(x, color='black', linewidth=lw)
            
            # Plot numbers
            for row in range(9):
                for col in range(9):
                    ax.text(col + 0.5, row + 0.5, str(grid[row, col]),
                            ha='center', va='center', fontsize=10, color='black')
                    
def plot_compare_puzzles(results, unsolved, solved, fig_title="compare_puzzles"):
    # Extract everything with their keys
    indexes = list(results.keys())
    gen_solution = [results[k]["solution"] for k in indexes]
    accuracies = [results[k]["accuracy"] for k in indexes]
    fitness_scores = [results[k]["fitness_score"] for k in indexes]
    pop_sizes = [results[k]["pop_size"] for k in indexes]
    generations = [results[k]["gen_size"] for k in indexes]
    times = [results[k]["seconds"] for k in indexes]

    # set up figure and grids
    num_plots = len(indexes)
    fig = plt.figure(figsize=(14, 4 * num_plots))
    gs = gridspec.GridSpec(num_plots, 3, width_ratios=[0.5, 1, 1], wspace=0.0001, hspace=0.3)

    for i in range(num_plots):
        # metrics
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis('off')
        metrics_text = (
            f"Puzzle Index: {indexes[i]}\n"
            f"Accuracy: {accuracies[i]:.3f}\n"
            f"Fitness Score: {fitness_scores[i]:.3f}\n"
            f"Population Size: {pop_sizes[i]}\n"
            f"Num Generations: {generations[i]}\n"
            f"Runtime: {times[i]:.3f} seconds"
        )
        ax_text.text(0, 0.5, metrics_text, ha='left', va='center', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        # middle grid
        ax_gen = fig.add_subplot(gs[i, 1])
        gen_grid = gen_solution[i]
        unsolved_grid = unsolved[i]
        solved_grid = solved[i]
        generated_vals = get_replacement_indexes(unsolved_grid)
        ax_gen.set_title("Generated Solution")
        ax_gen.set_xlim(0, 9)
        ax_gen.set_ylim(0, 9)
        ax_gen.invert_yaxis()
        ax_gen.set_xticks(np.arange(0, 10, 1))
        ax_gen.set_yticks(np.arange(0, 10, 1))
        ax_gen.set_aspect('equal')
        ax_gen.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        for x in range(10):
            lw = 2 if x in [0, 3, 6, 9] else 1
            ax_gen.axvline(x, color='black', linewidth=lw)
            ax_gen.axhline(x, color='black', linewidth=lw)

        # color code generated values
        for row in range(9):
            for col in range(9):
                val = gen_grid[row, col]
                if (row, col) in generated_vals:
                    if val == 0:
                        color = "#4d83db"
                    elif val != solved_grid[row, col]:
                        color = "red"
                    else:
                        color = "#0aa105"
                else:
                    color = "black"
                ax_gen.text(col + 0.5, row + 0.5, str(val), ha='center', va='center', fontsize=10, color=color)

        # right grid
        ax_solved = fig.add_subplot(gs[i, 2])
        draw_grid(ax_solved, solved_grid, "Solved Puzzle")

    plt.savefig(f"{fig_title}.png")


### FITNESS FUNCTIONS
def num_repeats(arr):
    """ counts the number of duplicate values in a row/column/subgrid

    Args:
        arr (numpy.ndarray): represents row/column/subgrid of a puzzle

    Returns: 
        len(repeats.values()) (int): total number of values that are repeated
        repeats (dictionary): number of times specific digits are repeateded
    """
    # set keys of dict to values in structure
    numbers = {value: 0 for value in arr}

    # count frequency of individual numbers
    for num in arr:
        numbers[num] += 1

    # count number of repeats
    repeats = {}
    for key, val in numbers.items():
        if key != 0:
            if val > 1:
                repeats[key] = val

    return len(repeats.keys()), repeats

def num_empty_cells(arr):
    """ counts the number of duplicate values in a row/column/subgrid

    Args:
        arr (numpy.ndarray): row/column/subgrid of a puzzle

    Returns: 
        len(zeros) (int): total number of values that are 0
    """
    # get list of values in structure that are equal to 0
    zeros = [x for x in arr if x==0]
    return len(zeros)

def get_fitness(generated_sol, weights=[0.8, 0.4]):
    """ calculates weighted fitness score for a puzzle

    Args:
        full_arr (numpy.ndarray): 9x9 puzzle TODO
        weights (list, optional): first value is the weight for number of repeats, 
            second is weight for number of empty cells. defaults to [0.8, 0.4]

    Returns:
        fitness (float): score representing how "good" a solution to a puzzle is. lower values mean more fit
    """

    # get number of repeats in rows, columns, and subgrids
    structures = ["row", "column", "subgrid"]
    repeats = {"row":0, "column":0, "subgrid":0}
    visited_subgrids = []
    for struct in structures:
        for idx in range(9):
            if struct == "subgrid":
                for j in [0, 3, 6]:
                    sub_x, sub_y = get_subgrid_coordinates(idx, j)
                    if (sub_x, sub_y) not in visited_subgrids:
                        arr = get_structure(generated_sol, struct, sub_x, sub_y)
                        repeats[struct] += num_repeats(arr)[0]
                        visited_subgrids.append((sub_x, sub_y))
            else:
                arr = get_structure(generated_sol, struct, idx)
                repeats[struct] += num_repeats(arr)[0]
    total_repeats = sum(repeats.values())

    # get total number of empty cells in puzzle
    empties = 0
    for idx in range(9):
        arr = get_structure(generated_sol, "row", idx)
        num_empty = num_empty_cells(arr)
        empties += num_empty

    # calculate fitness score
    fitness = weights[0]*total_repeats + weights[1]*empties

    return fitness

### FILLING METHODS
def strat_fill_remaining_vals(full_arr, structure, idx, sg_idx=0):
    """ strategically fill in remaining values in empty cells only if those values 
            do not exist in the corresponding row/column/subgrid
        resulting array will have 0s if the remaining values exist in corresponding row/column/subgrid

    Args: 
        full_arr (numpy.ndarray): 9x9 puzzle
        structure (string): row, column, or subgrid
        idx (int): starting index of structure
        sg_idx (int): y index of subgrid, optional, default=0

    Returns:
        arr (numpy.ndarray): row/column/subgrid of a puzzle with randomly filled remaining vals
    """
    # get desired sub array
    arr = get_structure(full_arr, structure, idx, sg_idx).copy()

    # get indexes of 0s
    zeros_ind = np.flatnonzero(arr == 0)
    np.random.shuffle(zeros_ind)

    invalid_placement_attempt = 0
    while len(zeros_ind) > 0:
        # get remaining values
        rem_vals = get_remaining_values(arr)
        np.random.shuffle(rem_vals)

        if invalid_placement_attempt > len(rem_vals)+10:
            break
            # TODO THIS BREAK STRATEGY COULD PREMATURELY STOP THE LOOP
    
        # choose random remaining value
        rand_remaining = int(np.random.choice(rem_vals))

        # choose random 0 index
        rand_zero_ind = int(np.random.choice(zeros_ind))

        # get corresponding row/column/subgrid arr
        if structure == "subgrid":
            corr_arr_1 = get_structure(full_arr, "row", rand_zero_ind)
            corr_arr_2 = get_structure(full_arr, "column", rand_zero_ind)
        else:
            sub_x, sub_y = get_subgrid_coordinates(idx, rand_zero_ind)
            corr_arr_1 = get_structure(full_arr, "subgrid", sub_x, sub_y)
            other_structure = [sub for sub in ["row", "column"] if sub != structure][0]
            corr_arr_2 = get_structure(full_arr, other_structure, rand_zero_ind)

        # check if random value is in corresponding structures
        if rand_remaining not in corr_arr_1 and rand_remaining not in corr_arr_2:
            # remove random remaining value from remaining values list
            rem_vals = rem_vals[rem_vals != rand_remaining]

            # remove that index from zeros index list
            zeros_ind = zeros_ind[zeros_ind != rand_zero_ind]
            
            # place remaining value at zero index
            arr[rand_zero_ind] = rand_remaining
        
        else:
            invalid_placement_attempt += 1

    return arr

## GENETIC ALGORITHM
def initialize_candidate(full_array):
    """ use filling method(s) to fill in empty values of an unsolved puzzle

    Args:
        full_arr (numpy.ndarray): 9x9 puzzle, no empty values filled
    
    Returns:
        full_arr (numpy.ndarray): 9x9 puzzle, some/most empty values filled
    """
    # copy input
    full_arr = np.copy(full_array)

    # randomly choose between filling rows, columns, or grids
    options = ["row", "column", "subgrid"]

    rc_indexes = [idx for idx in range(9)]
    sg_indexes = [0, 3, 6]

    # repeatedly find empty values to fill 
    for choices in range(len(options)):
        # randomly start filling by row/column/subgrid
        np.random.shuffle(options)
        structure = np.random.choice(options) 
        options.remove(structure)

        # fill all rows, columns, or subgrids
        if structure == "row" or structure == "column":
            np.random.shuffle(rc_indexes)
            for i in rc_indexes:
                filled = strat_fill_remaining_vals(full_arr, structure, i)
                full_arr = replace_structure(full_arr, filled, structure, i)
            
        elif structure == "subgrid":
            np.random.shuffle(sg_indexes)
            for i in sg_indexes:
                for j in sg_indexes:
                    filled = strat_fill_remaining_vals(full_arr, structure, i, j)
                    full_arr = replace_structure(full_arr, filled, structure, i, j)
    return full_arr

def selection(population, fitness_scores, num_select=5):
    """ chooses most fit variations of a puzzle TODO
    """
    # combine population with fitness scires
    pop_scores = {}
    for i, (arr, num) in enumerate(zip(population, fitness_scores)):
        pop_scores[i] = {"array": arr, "value": num}

    # sort by fitness scores in ascending order
    items_list = [(key, value) for key, value in pop_scores.items()]
    sorted_items = sorted(items_list, key=lambda x: x[1]["value"])

    sorted_dict = {}
    for new_idx, (old_idx, data) in enumerate(sorted_items):
        sorted_dict[new_idx] = data
        
    # selects top num_select populations
    first_n_keys = list(sorted_dict.keys())[:num_select]
    result_dict = {}
    for key in first_n_keys:
        result_dict[key] = sorted_dict[key]
    
    return result_dict

def pair_parents(parents):
    # create pairings of parents to be bred
    pairs = []
    for parent1 in parents:
        for parent2 in parents:
            if (parent1, parent2) not in pairs and (parent2, parent1) not in pairs:
                pairs.append((parent1, parent2))
    return pairs

def crossover_structure(parent1, parent2):
    # randomly choose row or column TODO ADD SUBGRID
    structure = np.random.choice(["row", "column"])

    # randomly choose number of structures to swap (1-9 inclusive)
    swaps = np.random.randint(1, 10)

    # get indexes of swaps
    possible_values = np.arange(0, 9) 
    indexes = np.random.choice(possible_values, size=swaps, replace=False)

    # get structures to swap
    from_1 = {}
    from_2 = {}
    for idx in indexes:
        from_1[idx] = get_structure(parent1, structure, idx)
        from_2[idx] = get_structure(parent2, structure, idx)
    
    # replace structures in parent1 with those from parent2 and vice versa
    for idx in indexes:
        parent1 = replace_structure(parent1, from_2[idx], structure, idx)
        parent2 = replace_structure(parent2, from_1[idx], structure, idx)

    return parent1, parent2
  
def replacement(population, offspring, pop_fitness_scores):
    """ TODO """
    # combine population with fitness scores
    pop_scores = {}
    for i, (arr, num) in enumerate(zip(population, pop_fitness_scores)):
        pop_scores[i] = {"array": arr, "value": num}

    # calculate fitness scores of offspring
    off_fitness_scores = [get_fitness(candidate) for candidate in offspring]
    
    # combine offspring with fitness scores
    for j, (arr, num) in enumerate(zip(offspring, off_fitness_scores)):
        pop_scores[i+j+1] = {"array": arr, "value": num}

    # find best populations
    pops = [element["array"] for element in pop_scores.values()]
    scores = [element["value"] for element in pop_scores.values()]
    best_pop_score = selection(pops, scores, int(len(pops)*0.3))
    best_populations = [element["array"] for element in best_pop_score.values()]

    return best_populations
        
def gen_hill_neighbors(puzzle, to_fill, num_neighbors=15):
    # of the og to fill,
    # check which have been filled
    # TODO - DO I CHANGE ONE VALUE THAT HAS ALREADY BEEN GENERATED OR FOCUS ON FILLING 0s
    # currently, replacing any random value
    neighbors = [puzzle]

    # randomly choose non-preset value to replace
    for i in range(num_neighbors):
        idx = np.random.choice(len(to_fill)-1)
        x, y = to_fill[idx]

        # get available values to replace this cell with
        not_in_row = set(get_remaining_values(get_structure(puzzle, "row", x)))
        not_in_col = set(get_remaining_values(get_structure(puzzle, "column", y)))
        not_in_subgrid = set(get_remaining_values(get_structure(puzzle, "subgrid", x, y)))
        valid_values = list(not_in_row.intersection(not_in_col, not_in_subgrid))

        # replace cell with randomly chosen valid value
        if len(valid_values) > 0:
            neighbor = puzzle.copy()
            neighbor[x][y] = np.random.choice(valid_values)
            neighbors.append(neighbor)

    return neighbors

def hill_climb(generated_sol, max_iters=1000, num_neighbors=50):
    current = generated_sol
    current_fitness = get_fitness(current)  # Your fitness function
    iteration = 0

    while iteration < max_iters:
        # Generate all neighbor candidates (or a subset, if too many)
        neighbors = gen_hill_neighbors(current, get_replacement_indexes(current), num_neighbors)
        
        # Find the neighbor with the best fitness
        best_neighbor = None
        best_fitness = current_fitness

        for neighbor in neighbors:
            neighbor_fitness = get_fitness(neighbor)
            if neighbor_fitness < best_fitness:
                best_neighbor = neighbor
                best_fitness = neighbor_fitness

        # If a better neighbor is found, move to it; otherwise, stop.
        if best_neighbor is not None:
            current = best_neighbor
            current_fitness = best_fitness
        else:
            break  # Local optimum reached

        iteration += 1

    return current


"""def all_arrays_equal(arr_list):
    if not arr_list: TODO
        return True  # empty list is trivially "equal"

    first = arr_list[0]
    return all(np.array_equal(first, arr) for arr in arr_list[1:])"""


def genetic_algorithm(unsolved, solved, population_size, generations, mutation_rate=0, hill_climbing=False, hill_iter=1000, num_neighbors=50):
    #TODO
    start = time.time()
    dELETE_TODO = 1

    # create population of solutions for one puzzle
    population = []
    for _ in range(population_size):

        candidate = initialize_candidate(unsolved)
        if isinstance(candidate, np.ndarray):
            population.append(candidate)

    for gen in range(generations):
        # TODO
        if gen % 10 == 0:
            print(gen, generations)

        # run hill climbing on each candidate
        if hill_climbing:
            population = [hill_climb(candidate, hill_iter, num_neighbors) for candidate in population]

        # calculate fitness for each candidate
        fitness_scores = [get_fitness(candidate) for candidate in population if isinstance(candidate, np.ndarray)]

        # check for perfect solution
        if 0 in fitness_scores:
            solution = population[fitness_scores.index(0)]
            return solution, gen
    
        # choose most fit populations (top 30%)
        parents = selection(population, fitness_scores, int(population_size*0.3))
        
        # produce offspring with crossover
        offspring = []
        for parent1_idx, parent2_idx in pair_parents(parents):
            parent1 = parents[parent1_idx]["array"]
            parent2 = parents[parent2_idx]["array"]
            child1, child2 = crossover_structure(parent1, parent2)
            offspring.extend([child1, child2])

        # randomly mutate TODO
        #offspring = [mutation(child, mutation_rate) for child in offspring]
        
        # form new generation
        if len(population) > 1:
            population = replacement(population, offspring, fitness_scores)
        #print("population:", population)
        
        # Optional: Check for plateau and invoke local search if needed TODO
        dELETE_TODO += 1

    # return the best candidate found
    # TODO RUN HILL CLIMBING AGAIN?
    best_sol_val = selection(population, fitness_scores, 1)[0]
    best_solution = best_sol_val["array"]
    best_fitness_score = best_sol_val["value"]
    end = time.time()
    print() #TODO
    return best_solution, best_fitness_score, end-start
    



def main():
    # load in data
    all_puzzles = pd.read_csv('sudoku_500.csv') 

    # convert strings of ints to 9x9 numpy arrays
    transformed = all_puzzles.map(lambda x: np.array([int(i) for i in list(x)]).reshape((9, 9)))

    # isolate unsolved puzzles and their solutions
    puzzles = transformed["puzzle"]
    solutions = transformed["solution"]


    # testing TODO
    """x = np.arange(100, 251, 10)  
    y_direct = x.copy()
    np.random.seed(42)
    y_random = np.random.permutation(x)
    population_sizes = np.concatenate([x, x])          
    generations = np.concatenate([y_direct, y_random])  
    puzzle = puzzles[5]
    solution = solutions[5]
    results = compare_parameters(puzzle, solution, population_sizes, generations)
    plot_compare_parameters(results)"""
    # choose 10 random puzzles to solve
    # define population and generation sizes
    indexes = [0, 2]    
    pop_sizes = [5, 7]
    gens = [3, 7]
    #arr = get_structure(puzzles[0], "row", 0)
    #print(get_remaining_values(arr))
    generated_vals = [[(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)], [(0, 0), (0, 1), (8, 0), (8, 8)]]
    test = puzzles[0]
    #print(test)

    #results = compare_puzzles(indexes, puzzles, solutions, pop_sizes, gens)
    #plot_compare_puzzles(results, puzzles, solutions, fig_title="test_compare_puzzles")
    #print(len(pop_sizes))
    # create population
    """population = [] # list of numpy arrays
    for _ in range(3):
        candidate = initialize_candidate(test)
        if isinstance(candidate, np.ndarray):
            population.append(candidate)
    print(population)
    print()"""

    indexes = [10, 10]
    pop_sizes = [10]#[150, 150]
    gens = [10] #[125, 125]
    hills = [False, True]
    pop_gen_idx = 0
    unsolved = puzzles[indexes[pop_gen_idx]]
    solved = solutions[indexes[pop_gen_idx]]
    sol, fitness_score, time = genetic_algorithm(unsolved, solved, pop_sizes[pop_gen_idx], gens[pop_gen_idx], hill_climbing=hills[pop_gen_idx])
    print("RESUTLS:")
    print("unsolved:")
    print(unsolved)
    print("solved:")
    print(solved)
    print("generated solution:")
    print(sol)
    print("fitness:", fitness_score)
    print("time:", time)
    print("accuracy:", get_accuracy(unsolved, solved, sol))
    #results = compare_puzzles(indexes, puzzles, solutions, pop_sizes, gens, hills)
    #plot_compare_puzzles(results, puzzles, solutions, fig_title="no_hill_VS_hill")



# TODO - ACCURACY SHOULD NOT BE INCLUDED IN FITNESS CALCULATION


    """NEXT STEPS
X Representation of the Candidate Solution
    use strat_fill_remaining_vals to fill a whole puzzle

X Fitness Function Design
    Use your num_repeats function to count duplicates per row/column/subgrid.
    Optionally incorporate num_empty_cells if using a representation that can leave some cells unfilled.
    Define an overall fitness score (lower is better) that aggregates these penalties.

X Initial Population Generation
    steps:
        Loop over the number of individuals in your desired population.
        For each, copy the initial puzzle.
        Apply a fill method (e.g., your strategic filling function) that respects fixed values and randomly fills the rest.
        Outcome: A diverse set of candidate solutions to begin the evolutionary process.

X Selection Mechanism
    Choose parents for reproduction based on fitness
    Tournament Selection: Randomly pick a subset of individuals and choose the best among them.
    Roulette Wheel Selection: Assign probabilities proportional to fitness (or inversely, if lower fitness means better)
    Create a function that takes the population and returns selected parents for crossover.

X Crossover (Recombination)
    Combine parts of two parent solutions to produce offspring.
    Row/Column/Subgrid Crossover: Exchange entire rows, columns, or subgrids between parents while preserving fixed cells.
    Block-based Crossover: For example, randomly select a block (a set of contiguous rows) to swap
    Ensure that the resulting offspring still adhere to the fixed cell constraints.

TODO Mutation
    Introduce diversity by making small random changes.
    Randomly select a mutable cell and assign it a new value (from 1 to 9) that does not immediately violate fixed constraints.
    Alternatively, swap two mutable values in a row, column, or subgrid.
    Create a mutation function that is applied with a given probability per individual or per cell.

X Generation Update 
    Form a new generation by replacing some or all of the old population with offspring.
    Consider elitism: retain a few of the best individuals unchanged to ensure good solutions aren’t lost.
    loop:
        Evaluate fitness for all individuals.
        Select parents.
        Generate offspring through crossover and mutation.
        Replace or merge populations

X Termination Criteria
    Decide When to Stop:
        A solution with a fitness score of zero (no violations) is found.
        A maximum number of generations or time limit is reached.
    Monitor if improvements plateau, in which case you might restart or introduce a local search component (Hill Climbing) to refine near-optimal candidates.

Integrating Local Search (Optional Enhancement)
    Fallback Strategy:
        If the genetic algorithm stops making progress, use a Hill Climbing algorithm:
        Apply local mutations that only make changes that improve the fitness.
        Restart with a different seed if no progress is made after several iterations.
    Help overcome plateaus or refine solutions that are close to valid.

Parameter Tuning and Evaluation
    experimentation:
        Vary population size, mutation rate, crossover strategy, and selection pressure.
        Compare performance metrics such as convergence speed, success rate (finding a valid solution), and resource (time/memory) consumption.
    comparison: Benchmark against alternative methods (like CSP-based approaches) to see the trade-offs in complexity and performance across varying puzzle difficulties.
    """

if __name__ == "__main__":
    main()

