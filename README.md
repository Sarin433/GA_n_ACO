# GeneticAnts
This GitHub repository is my interest in meta-heuristics to solve optimization problems that are NP-hard problems. 
- This repositories interest in TSP problems, ACO and GA

---
## Genetic Alogorithm
A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

From *[Medium by Vijini Mallawaarachchi](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3)*

---
## Ant Colony Optimization
Ant colony optimization (ACO) is an optimization algorithm which employs the probabilistic technique and is used for solving computational problems and finding the optimal path with the help of graphs.
    
From *[Applications of Big Data in Healthcare, 2021](https://www.sciencedirect.com/science/article/pii/B9780128202036000023)*

---
## Pseudocode of Hybrid ACO and GA
```bash
Define: num_Ants, num_iterations, alpha, beta
# Start the Algorithm
while iteration < num_iterations
do     
    start Ant by num_Ants to find path
        evaluate Ant
        Binary Tournament to select 2 Ants 
        do operation
            Order crossover 
            Inversion  Mutation  
        evaluate 'new_Ant' from  operation
        update pheromone by alpha abd beta
if iteration >= num_iterations
    break
```
---

# Referrence
- Ant Coloy Optimization 
from *[QiuBingCheng Github repositories](https://github.com/QiuBingCheng/MediumArticle)*