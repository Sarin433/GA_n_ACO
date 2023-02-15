# GeneticAnts
## Genetic Alogorithm
    A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

From *[Medium by Vijini Mallawaarachchi](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3)*

---
## Ant Colony Optimization
    Ant colony optimization (ACO) is an optimization algorithm which employs the probabilistic technique and is used for solving computational problems and finding the optimal path with the help of graphs.
    
From *[Applications of Big Data in Healthcare, 2021](https://www.sciencedirect.com/science/article/pii/B9780128202036000023)*

---
## Pseudocode
```bash
Define: num_Ants, num_iterations, mutation_rate
# Start the Algorithm
while iteration < num_iterations
do     
    start Ant by num_Ants to find path
        evaluate Ant
        select 'Best_Ant' and random select 'Ant' from Ant pool in iteration
        do 
            crossover operation between 'Best_Ant' and 'Ant' 
            mutation operation by mutation_rate
        evaluate 'new_Ant' from  operation
if iteration >= num_iterations
    break
```
---

