import numpy as np
import random
from copy import copy

class Bidder(object):

    def __init__(self, budget,num_items, num_rounds, lower=10, upper = 50):
        self.budget = budget
        self.private_eval = np.random.uniform(lower,upper, size=(num_items))

        #self.strategy = np.random.randint(low=0, high=2, size=(num_items, num_rounds))
        self.strategy = np.ones((num_items, num_rounds))
        self.num_items  = num_items
        self.num_rounds = num_rounds
        self.fitness = 0


    def fitness_eval(self, p):

        f = np.sum(self.private_eval[p > 0] - p[p > 0])
        if np.sum(p) > self.budget:
            self.fitness = self.budget - np.sum(p)
        else:
            self.fitness =  f
        return self.fitness


def english_auction(bidders, delta, initial_price):

    num_items = len(bidders[0].private_eval)
    num_bidders = len(bidders)

    p_mat = np.zeros((num_bidders, num_items))
    for it in range(num_items):

        highest_round = [None] * num_bidders
        for idx, bidder in enumerate(bidders):
            highest_round[idx] = np.argmax(bidder.strategy[:,it] == 0)

            #if highest_round is still None then it means that the player has
            #a strategy of bidding no matter what
            if highest_round[idx] == None:
                highest_round[idx] = bidder.strategy.shape[0]

        highest_bidder = highest_round.index(max(highest_round))
        highest_round.sort(reverse=True)

        p_mat[highest_bidder,it] = initial_price + delta*(highest_round[1] + 1)

    return p_mat

def tournament_select(pop, p_tour, tour_size):
    iSelected = None


    if tour_size > len(pop):
        return None
    #chose N individuals randomly and sort them by fitness
    selected_ind = [random.choice(pop) for _ in range(tour_size)]
    selected_ind.sort(key=lambda x:x.fitness, reverse=True)


    while iSelected == None:
        r = random.uniform(0,1)

        #chose best individual with probabilty p_tour
        if r < p_tour:
            iSelected = selected_ind[0]
        #if there are only 2 ind left then choose the second one with prob 1 - p_tour
        elif len(selected_ind) == 2:
            iSelected = selected_ind[1]
        #otherwise remove the best ind from tournament
        else:
            selected_ind.pop(0)
    return iSelected

def crossover(p1, p2, type='single'):

    if type == 'single':
        num_genes = p1.strategy.shape[1]
        num_items = p1.strategy.shape[0]
        crosspoint = np.random.randint(low=1, high=num_genes*num_items)

        ##flatten the matrix for crossoveer op
        p1_chromosome = p1.strategy.ravel()
        p2_chromosome = p2.strategy.ravel()


        tmp = p1_chromosome[:crosspoint].copy()

        p1_chromosome[:crosspoint] = p2_chromosome[:crosspoint]
        p2_chromosome[:crosspoint] = tmp


        p1.strategy = p1_chromosome.reshape(num_items, num_genes)
        p2.strategy = p2_chromosome.reshape(num_items, num_genes)


    elif type == 'multiple':
        pass

def mutate_strategy(individual, p_mut):
    num_items = individual.num_items
    num_rounds = individual.num_rounds

    for it in range(num_items):
        for jt in range(num_rounds):
            r = np.random.uniform()
            if r < p_mut:
                individual.strategy[it, jt] = 1 - individual.strategy[it,jt]


def main(num_items, num_rounds, num_players):
    generations = 100

    budget = 2*100
    p_tour = 0.75
    tour_size = 2
    p_mut = 0.05
    p_cossover = 0

    #Create players/bidders
    bidders = []
    for idx in range(num_players):
        b = Bidder(budget, num_items, num_rounds)
        bidders.append(b)


    #For each generation do:
    for iGen in range(generations):

        best_strategy = bidders[0].strategy.copy()
        best_bidder_id = 0
        best_ind = bidders[0]
        #perform auction
        p = english_auction(bidders, 10, 10)

        #calculate the payoff and fitness for each bidder
        #save a copy of the best bidder
        for idx, bidder in enumerate(bidders):
            idx_fitness = bidder.fitness_eval(p[idx,:])

            if bidder.fitness > best_ind.fitness:
                best_ind = bidder
                best_bidder_id = idx
                best_strategy = bidder.strategy.copy()

        #Tournament selection and crossover
        for i in range(0, num_players):
            i1 = copy(bidders[i])
            mutate_strategy(bidders[i], p_mut)
            #i2 = tournament_select([i1, bidders[i] ], p_tour, tour_size)
            p = english_auction(bidders, 10, 10)
            f = bidders[i].fitness_eval(p[i,:])

            #if np.random.uniform() < p_tour:
            if  i1.fitness > f:
                bidders[i] = i1


            if random.uniform(0, 1) < p_cossover:
                crossover(i1, i2)       #done in place
            else:
                pass

        #mutations
    #    for bidder in bidders:
    #        mutate_strategy(bidder, p_mut)

        #Elitism
        bidders[best_bidder_id].strategy = best_strategy

        print(best_ind.fitness)
        print(best_ind.private_eval)
        print(best_ind.strategy)
        print("----")
    print(p)
if __name__ == "__main__":
    main(5, 100, 20)
