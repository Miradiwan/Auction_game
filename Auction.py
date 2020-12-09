import numpy as np
import random


class Bidder(object):

    def __init__(self, budget,num_items, num_rounds, lower=1, upper = 100):
        self.budget = budget
        self.private_eval = np.random.uniform(lower,upper, size=(num_items))

        self.strategy = np.random.randint(low=0, high=2, size=(num_items, num_rounds))

        self.num_items  = num_items
        self.num_rounds = num_rounds
        self.fitness = 0


    def fitness_eval(self, p):

        f = np.sum(self.private_eval[p > 0] - p[p > 0])
        if np.sum(p) > self.budget:
            self.fitness = float(-inf)
        else:
            self.fitness =  f



def english_auction(bidders, delta, initial_price):

    num_items = len(bidders[0].private_eval)
    num_bidders = len(bidders)

    p_mat = np.zeros((num_bidders, num_items))
    for it in range(num_items):
        highest_round = 0
        second_highest_round = None
        highest_round_bidder = 0

        for idx, bidder in enumerate(bidders):
            last_round = np.argmax(bidder.strategy[:,it] == 0)

            if last_round > highest_round:
                second_highest_round = highest_round
                highest_round = last_round
                highest_round_bidder = idx

        if second_highest_round:
            p_mat[highest_round_bidder,it] = initial_price + delta*(second_highest_round + 1)

    return p_mat

def tournament_select(pop, p_tour, tour_size):
    iSelected = None

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
            if random.uniform(0, 1) > p_mut:
                individual.strategy[it, jt] = 1 - individual.strategy[it,jt]


def main(num_items, num_rounds, num_players):
    generations = 5000

    p_tour = 0.75
    tour_size = 5
    p_mut = 0.2
    p_cossover = 0.7

    #Create players/bidders
    bidders = []
    for idx in range(num_players):
        b = Bidder(1000, num_items, num_rounds)
        bidders.append(b)

    #For each generation do:
    for iGen in range(generations):

        #perform auction
        p = english_auction(bidders, 10, 10)

        best_ind = bidders[0]
        #calculate the payoff and fitness for each bidder
        #save a copy of the best bidder
        for idx, bidder in enumerate(bidders):
            bidder.fitness_eval(p[idx,:])

            if bidder.fitness > best_ind.fitness:
                best_ind = bidder

        #Tournament selection and crossover
        for i in range(0, num_players, 2):
            i1 = tournament_select(bidders, p_tour, tour_size)
            i2 = tournament_select(bidders, p_tour, tour_size)

            if random.uniform(0, 1) > p_cossover:
                crossover(i1, i2)       #done in place

        #mutations
        #for bidder in bidders:
        #    mutate_strategy(bidder, p_mut)

        for bidder in bidders:
            print(bidder.private_eval)
        #print(best_ind.strategy)
        print("----")
if __name__ == "__main__":
    main(5, 10, 10)
