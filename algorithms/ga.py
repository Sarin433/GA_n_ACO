import random 
import numpy as np
import sys

class GeneticAlgorithm:
  def __init__(self, city_list, pop_size, cx_rate, mut_rate, elite, obj_function):
        self.city_list = city_list
        self.pop_size = pop_size
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.elite = elite
        self.obj_function = obj_function

  def create_route(self):    
        return np.random.permutation(self.city_list).tolist()

  def initial_population(self):
        population = []
        for i in range(0, self.pop_size):
            population.append(self.create_route())            
        return sorted(population, key=lambda population: self.obj_function(population))

  def initialize(self):
        self.best_objective_value = sys.float_info.max
        self.best_solution = []
        self.population = self.initial_population()
        self.obj_value = np.zeros(self.pop_size)
        self.solutions = np.zeros((self.pop_size,len(self.city_list)),dtype=int)

  def sel_tournament(self, pool):
        chosen = []
        
        p1 = random.choice(pool)
        p2 = random.choice(pool)  
        winner = min([p1, p2], key = lambda x: self.obj_function(x))
        chosen = winner 
          
        return chosen
        
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
  
  def create_next_gen(self):
        previous_population = self.population.copy()
        elite_pop = previous_population[:self.elite]
        for i in range(self.pop_size - self.elite):
            parent1 = self.sel_tournament(self.population)
            parent2 = self.sel_tournament(self.population)
            # CX rate
            #  off1, off2 = self.order_crossover(parent1, parent2) if random.random <= self.cx_rate else parent1, parent2
            off1, off2 = self.order_crossover(parent1, parent2)
            off1 = self.mut_inversion(off1) if random.random() <= self.mut_rate else off1
            # if random.random() <= self.mut_rate:
            #     new_off1 = self.mut_inversion(off1)
            off2 = self.mut_inversion(off2) if random.random() <= self.mut_rate else off2
            # if random.random() <= self.mut_rate:
            #     new_off2 = self.mut_inversion(off2)
            sel_off = min([off1, off2], key = lambda x: self.obj_function(x))
            elite_pop.append(sel_off)
        
        next_population = sorted(elite_pop, key=lambda elite_pop: self.obj_function(elite_pop))
        # Update Next Pop
        self.population = next_population.copy()
        # Update each Objective Value
        for i in range(len(next_population)):
            self.obj_value[i] = self.obj_function(next_population[i])
            self.solutions[i] = next_population[i]

        
  
  def update_best_solution(self):
        for i, value in enumerate(self.obj_value):
            if (value < self.best_objective_value):
                self.best_objective_value = value
                self.best_solution = self.solutions[i].copy()
