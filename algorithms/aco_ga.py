import random
import numpy as np
import sys

class Graph:
    def __init__(self, nodes, location):
        self.nodes = nodes 
        self.location = location
        self.pheromone = [[1/(len(nodes) * len(nodes)) for j in range(len(nodes)) for i in range(len(nodes))]]

    def create_distance_matrix(self):
        x, y = np.array(self.location).T
        dist = np.sqrt((x[:, np.newaxis] - x)**2 + (y[:, np.newaxis] - y)**2)
        np.fill_diagonal(dist, 0) # set diagonal to 0
        
        return dist.tolist()
    
    def TSPProblem_obj(self, path):
        total_dist = 0
        self.distance_matrix = self.create_distance_matrix()
        total_dist += self.distance_matrix[path[-1]][path[0]]
        for i in range(len(path) - 1):
            total_dist += self.distance_matrix[path[i]][path[i+1]]

        return total_dist
    
class GeneticAnts:
    def __init__(self, pop_size, location, distance_matrix, alpha, beta, pheromone_drop_amount, evaporate_rate, compute_obj_value):
        self.num_ants = pop_size
        self.location = location
        self.distance_matrix = distance_matrix
        self.alpha = alpha
        self.beta = beta
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.compute_obj_value = compute_obj_value

    def initialize(self):
        self.best_objective_value = sys.float_info.max
        self.obj_value = np.zeros(self.num_ants)
        self.best_solution = []
        self.solutions = np.zeros((self.num_ants,len(self.location)),dtype=int)
        self.pheromone_map = np.ones((len(self.location), len(self.location)))
        self.visibility = np.zeros((len(self.location), len(self.location)))
        for from_ in range(len(self.location)):
            for _to in range(len(self.location)):
                if (from_ == _to):continue
                distance = self.distance_matrix[from_][_to]
                try:self.visibility[from_][_to] = 1/distance
                except ZeroDivisionError:continue
        
    
    def do_roulette_wheel_selection(self, fitness_list):
        sum_fitness = sum(fitness_list)
        transition_probability = [fitness/sum_fitness for fitness in fitness_list]
    
        rand = random.random()
        sum_prob = 0
        for i,prob in enumerate(transition_probability):
            sum_prob += prob
            if(sum_prob>=rand):
                return i
    
    def create_ant_solution(self, location):
        nodes_list = [i for i in range(len(location))]
        start_city_id = random.choice(nodes_list)
        # solution = []
        solution = [start_city_id]
        nodes_list.remove(start_city_id)
        for i in range(1, len(location)-1):
            fitness_list = []
            for city_id in nodes_list:
                update_alpha = pow(self.pheromone_map[start_city_id][city_id], self.alpha)
                update_beta = pow(self.visibility[start_city_id][city_id], self.beta)
                fitness = update_alpha * update_beta
                fitness_list.append(fitness)

            next_city_id = nodes_list[self.do_roulette_wheel_selection(fitness_list)]
            nodes_list.remove(next_city_id)
            solution.append(next_city_id)

            start_city_id = next_city_id
        # nodes_list.pop()
        last_node = nodes_list.pop()
        solution.append(last_node)

        # print(solution,"\n",nodes_list)
        return solution
    
    def sel_random(self, pool):
        return random.choice(pool)
    
    def order_crossover(self, parent1, parent2):
        size = min(len(parent1), len(parent2))

        off1, off2 = [-1] * size, [-1] * size
        start, end = sorted([random.randrange(size) for _ in range(2)])

        off1_inherited = []
        off2_inherited = []
        for i in range(start, end+1):
            off1[i] = parent1[i]
            off2[i] = parent2[i]
            off1_inherited.append(parent1[i])
            off2_inherited.append(parent2[i])

        current_parent1_position, current_parent2_position = 0, 0

        fixed_pos = list(range(start, end + 1))       
        i = 0
        while i < size:
            if i in fixed_pos:
                i += 1
                continue

            test_off1 = off1[i]
            if test_off1 ==-1: #to be filled
                parent2_trait = parent2[current_parent2_position]
                while parent2_trait in off1_inherited:
                    current_parent2_position += 1
                    parent2_trait = parent2[current_parent2_position]
                off1[i] = parent2_trait
                off1_inherited.append(parent2_trait)

            test_off2 = off2[i]
            if test_off2 ==-1: #to be filled
                parent1_trait = parent1[current_parent1_position]
                while parent1_trait in off2_inherited:
                    current_parent1_position += 1
                    parent1_trait = parent1[current_parent1_position]
                off2[i] = parent1_trait
                off2_inherited.append(parent1_trait)
            i +=1

        return off1, off2
        
    def sel_tournament(self, pool):
        chosen = []
        
        p1 = random.choice(pool)
        p2 = random.choice(pool)  
        winner = min([p1, p2], key = lambda x: self.compute_obj_value(x))
        chosen = winner 
          
        return chosen
    
    def mut_inversion(self,individual):
    
        size = len(individual)
        if size == 0:
            return individual,

        index_one = random.randrange(size)
        index_two = random.randrange(size)
        start_index = min(index_one, index_two)
        end_index = max(index_one, index_two)

        # Reverse the contents of the individual between the indices
        individual[start_index:end_index] = individual[start_index:end_index][::-1]

        return individual

    def create_ant_pool(self):
        ants_pool = []
        for i in range(self.num_ants):
            # _ant = self.create_ant_solution(self.location)
            ants_pool.append(self.create_ant_solution(self.location))

        new_ants_pool = ants_pool.copy()
        for i in range(len(ants_pool)):
            parent1 = self.sel_tournament(ants_pool)
            parent2 = self.sel_tournament(ants_pool)
            off1, off2 = self.order_crossover(parent1, parent2)
            off1 = self.mut_inversion(off1)
            off2 = self.mut_inversion(off2)
            sel_off = min([off1, off2], key = lambda x: self.compute_obj_value(x))
            # mut_sel_off = self.mut_inversion(sel_off)
            new_ants_pool.append(sel_off)
            
        sorted_ants_pool = sorted(new_ants_pool, key=lambda new_ants_pool: self.compute_obj_value(new_ants_pool))

        for i in range(len(sorted_ants_pool[:self.num_ants])):
            self.obj_value[i] = self.compute_obj_value(sorted_ants_pool[i])
            self.solutions[i] = sorted_ants_pool[i]
    
    def update_pheromone(self):
        self.pheromone_map *= (1 - self.evaporate_rate)
        for solution in self.solutions:
            for j in range(len(solution)):
                node1 = solution[j]
                node2 = solution[j+1] if j<int(len(solution)-1) else solution[0]
                self.pheromone_map[node1, node2] += self.pheromone_drop_amount
        return self.pheromone_map
    
    def update_best_solution(self):
        for i, value in enumerate(self.obj_value):
            if (value < self.best_objective_value):
                self.best_objective_value = value
                self.best_solution = self.solutions[i].copy()
                # for n in range(self.num_ants):
                #     self.best_solution = self.solution[i][n]
                
                # self.best_obj_value = value