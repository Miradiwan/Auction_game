import numpy as np
import random
from string import ascii_uppercase
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
    # TODO:
    Add Dutch auction
    Add multiple round logic
    Change object for each round ??
'''
class Bidder(object):

    def __init__(self, budget, p_irrationality=0.5):

        self.irrationality = np.random.uniform(-p_irrationality, p_irrationality)
        self.budget = budget

        self.name = self._generate_id()

        self.num_items_bought = 0

    def roulette(self, eval_items, available_budget):
        chosen_item = []

        priv_items = self.prioritize_item(eval_items)

        prioritized_item = sorted(priv_items.items(), key=lambda kv: kv[1], reverse=True)

        while available_budget > 0 and len(chosen_item) != len(self.private_eval):

            if self.check_budget(available_budget, eval_items) == False:
                break

            s = 0
            r = np.random.uniform()


            for item in prioritized_item:
                k, v = item
                s += v

                if r < s:

                    if available_budget - self.private_eval[k] < 0:
                        break

                    #print("prob r : {:.2f} s : {:.2f} v : {:.2f}".format(r,s,v))
                    eval_items.pop(k, None)
                    priv_items = self.prioritize_item(eval_items)

                    prioritized_item = sorted(priv_items.items(), key = lambda kv: kv[1], reverse=True)

                    chosen_item.append(k)
                    available_budget -= self.private_eval[k]
                    break

        return chosen_item

    def check_budget(self, available_budget, priv_items):


        for k,v in priv_items.items():

            if available_budget > v:
                return True

        return False

    def prioritize_item(self, priv_eval):
        #Idea: to prioritize items with higher values
        d = sum( priv_eval.values() )

        item_prio = dict()
        for k, v in priv_eval.items():
            item_prio[k] = v/d

        return item_prio

    def make_bid(self, item_name, asking_price):

        if item_name not in self.chosen_items:
            return False
        if asking_price < self.private_eval[item_name]*(1 + self.irrationality):
            return True
        else:
            return False

    def set_private_eval(self, auction_items, lower_bound = 20, upper_bound = 150):

        self.private_eval = dict()

        for it in auction_items:
            self.private_eval[it.name] = np.random.uniform(lower_bound, upper_bound)

        self.chosen_items = self.roulette(self.private_eval.copy(), self.budget)

    def __repr__(self):
        return repr((self.irrationality, self.budget))

    def _generate_id(self, n=6  , chars=ascii_uppercase):
        return "".join(random.choice(chars) for i in range(n))


class Auction_Items(object):

    def __init__(self):

        self.name       = self._generate_id()
        self.sold_to    = None      # indicates which biddeer bought the item
        self.solf_for   = None      # The price the item was sold for
        self.highest_eval = 0       # The maximal evaluation of this object by any bidder
        self.second_highest_eval = 0

    def __repr__(self):
        return repr((self.name, self.sold_to, self.solf_for, self.highest_eval))

    def _generate_id(self, n=3, chars=ascii_uppercase):
        s = "".join(random.choice(chars) for i in range(n))
        return s

    def get_best_eval(self, bidders):
        bidders.sort(key=lambda x : x.private_eval[self.name], reverse=True)

        self.highest_eval = bidders[0].private_eval[self.name]
        self.second_highest_eval = bidders[1].private_eval[self.name]



def english_auction(bidders, Auction_Items, initial_bid, incremental_bid):

    for it in Auction_Items:
        still_bidding = bidders.copy()
        asking_price = initial_bid
        while not it.sold_to:

            for bidder in still_bidding:
                bid = bidder.make_bid(it.name, asking_price)

                if bid == 0:
                    still_bidding.remove(bidder)

                if len(still_bidding) == 1:
                    it.sold_to  = still_bidding[0]
                    it.sold_for = asking_price

                    payoff = still_bidding[0].private_eval[it.name] - asking_price
                    still_bidding[0].budget += payoff

                    still_bidding[0].num_items_bought += 1
                    break


            asking_price += incremental_bid



def main(num_items, num_bidders, budget):

    num_iter = 100
    initial_bid = 10
    incremental_bid = 1


    bidders = [Bidder(budget) for _ in range(num_bidders)]

    fig = plt.figure()

    for i in range(num_iter):

        #Generate new items
        auc_items = [Auction_Items() for _ in range(num_items)]

        #For each bidder set private eval:
        for bidder in bidders:
            bidder.set_private_eval(auc_items)

        #For each item find best evaluation
        for it in auc_items:
            it.get_best_eval(bidders)

        #For each bidder generate a lsit of item to bid on
        for bidder in bidders:
            bidder.chosen_item = bidder.roulette(bidder.private_eval.copy(), bidder.budget)


        #Play English auction
        english_auction(bidders, auc_items, initial_bid, incremental_bid)

        #reset budget:
        for bidder in bidders:
            print("\t {} \t{:10.2f} \t{:4.2f} \t{}".format(bidder.name, bidder.budget,
                bidder.irrationality, bidder.num_items_bought))


        for bidder in bidders:
            plt.plot(i, bidder.budget)#, label=bidder.irrationality)

        plt.pause(0.05)
        #plt.legend()
        print("-------------------")
    plt.show()
if __name__ == "__main__":
    num_bidders = 20
    num_items = 5
    budget = 350

    main(num_items, num_bidders, budget)
