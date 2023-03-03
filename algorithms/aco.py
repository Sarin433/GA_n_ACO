# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:43:07 2021
@author: Jerry
https://github.com/QiuBingCheng/MediumArticle
"""

import numpy as np
import sys
import random

class TSPProblem:
    def __init__(self,coordinate,cities_name):
        self.coordinate = coordinate
        self.cities_name = cities_name
    
    def get_distance(self,arr1,arr2):
        # Euclidean distance
        # return np.sqrt(np.power(arr1-arr2,2).sum())
        # return np.linalg.norm(arr1 - arr2)
        squares = [(p-q) ** 2 for p, q in zip(arr1, arr2)]
        return sum(squares) ** .5
    
    def compute_objective_value(self,cities_id):
        total_distance = 0
        for i in range(len(cities_id)):
            city1 = cities_id[i]
            city2 = cities_id[i+1] if i<len(cities_id)-1 else cities_id[0]
            total_distance += self.get_distance(self.coordinate[city1],self.coordinate[city2])
        return total_distance
    
    def to_cities_name(self,cities_id):
        return [self.cities_name[i] for i in cities_id]
        
class AntSystem:
    def __init__(self,pop_size,coordinate,pheromone_drop_amount,evaporate_rate,
                 pheromone_factor,heuristic_factor,
                 get_distance,compute_objective_value):
        
        self.num_ants = pop_size
        self.coordinate = coordinate
        self.num_cities = len(coordinate)
        self.get_distance = get_distance
        self.compute_objective_value = compute_objective_value
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        self.visibility_factor = heuristic_factor
        
    def initialize(self):
        self.one_solution = np.arange(self.num_cities,dtype=int)
        self.solutions  = np.zeros((self.num_ants,self.num_cities),dtype=int)
        for i in range(self.num_ants):
            for c in range(self.num_cities):
                self.solutions[i][c] = c
                
        self.objective_value = np.zeros(self.num_ants)
        self.best_solution = np.zeros(self.num_cities,dtype=int)
        self.best_objective_value = sys.float_info.max
        
        self.visibility = np.zeros((self.num_cities,self.num_cities))
        self.pheromone_map = np.ones((self.num_cities,self.num_cities))
        
        #heuristic_values
        for from_  in range(self.num_cities):
            for to in range(self.num_cities):
                if(from_==to):continue
                distance = self.get_distance(self.coordinate[from_],self.coordinate[to])
                try:self.visibility[from_][to] = 1/distance
                except ZeroDivisionError:continue
     
    def do_roulette_wheel_selection(self,fitness_list):
        sum_fitness = sum(fitness_list)
        transition_probability = [fitness/sum_fitness for fitness in fitness_list]
        
        rand = random.random()
        sum_prob = 0
        for i,prob in enumerate(transition_probability):
            sum_prob += prob
            if(sum_prob>=rand):
               return i
     
    def update_pheromone(self):
        #evaporate hormones all the path
        self.pheromone_map *= (1-self.evaporate_rate)
                
        #Add hormones to the path of the ants
        for solution in self.solutions:
            for j in range(self.num_cities):
                city1 = solution[j]
                city2 = solution[j+1] if j<self.num_cities-1 else solution[0]
                self.pheromone_map[city1,city2] += self.pheromone_drop_amount
            
    def _an_ant_construct_its_solution(self):
        candidates = [i for i in range(self.num_cities)]
        #random choose city as first city 
        current_city_id = random.choice(candidates)
        self.one_solution[0] = current_city_id
        candidates.remove(current_city_id)
        
        #select best from candiate
        for t in range(1,self.num_cities-1):
            #best
            fitness_list = []
            for city_id in candidates:
                fitness = pow(self.pheromone_map[current_city_id][city_id],self.pheromone_factor)*\
                    pow(self.visibility[current_city_id][city_id],self.visibility_factor)
                fitness_list.append(fitness)
            
            next_city_id = candidates[self.do_roulette_wheel_selection(fitness_list)]
            candidates.remove(next_city_id)
            self.one_solution[t] = next_city_id
            
            current_city_id = next_city_id
        self.one_solution[-1] = candidates.pop()

    def each_ant_construct_its_solution(self):
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            # print(i)
            for c in range(self.num_cities):  
                self.solutions[i][c] = self.one_solution[c] 
                
            self.objective_value[i] = self.compute_objective_value(self.solutions[i])
            # print(self.solutions[i])
            # print(self.objective_value[i])
                
    
    def update_best_solution(self):
        for i,val in enumerate(self.objective_value):
            if(val<self.best_objective_value):
                for n in range(self.num_cities):
                    self.best_solution[n] = self.solutions[i][n]
                
                self.best_objective_value = val