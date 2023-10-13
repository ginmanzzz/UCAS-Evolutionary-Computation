import random 
from network import netIndividual

def random_int_list(start, stop, length):
	# generate a random array
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def generateIndividual() -> dict :
    gene = {}
    gene['nb_node'] = random_int_list(16, 128, 3)
    gene['lr'] = random.uniform(0.001, 0.005)
    gene['batch_size'] = random.randint(16, 512)
    gene['kernels'] = random.choice([[3,3], [3,5], [5, 3], [5, 5]])
    gene['fitness'] = 0.0
    return gene

def generateGeneration(n:int):
    """
    Generate a generation with n individuals randomly
    """
    generation = []
    for i in range(n):
        generation.append(generateIndividual())
    return generation

def breed(father, mother):
    """
    Two individuals breead a baby with their gene
    """
    baby = {}
    for key in father:
        baby[key] = random.choice([father[key], mother[key]])
    return baby

def mutate(baby, mutate_prob):
    randomGene = generateIndividual()
    for key in randomGene:
        if random.random() < mutate_prob and key != 'fitness':
            baby[key] = randomGene[key]
    return baby

def evolve(generation, survive_percent, mutate_prob, isLast):
    origin_len = len(generation)
    survived_len = int(len(generation) * survive_percent)
    generation = sorted(generation, key = lambda individual: individual['fitness'], reverse=True)
    survived_generation = generation
    if isLast == False:
        survived_generation = generation[:survived_len] 
        while len(survived_generation) < origin_len:
            fatherID = random.randint(0, survived_len-1)
            motherID = random.randint(0, survived_len-1)
            baby = breed(survived_generation[fatherID], survived_generation[motherID])
            baby = mutate(baby, mutate_prob)
            survived_generation.append(baby)
    return survived_generation

def createGroup(generation):
    group = []
    for gene in generation:
        individual = netIndividual(gene)
        group.append(individual)
    return group

def main():
    generation = generateGeneration(10)
    print("generation")
    for gene in generation:
        print(gene)
    print("evolution")
    generation = evolve(generation, 0.4, 0.1, True)
    for gene in generation:
        print(gene)

    return
if __name__ == '__main__':
    main()
