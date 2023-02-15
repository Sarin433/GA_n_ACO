# GeneticAnts
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

