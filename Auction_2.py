import numpy as np
import random
from string import ascii_uppercase

class Bidder(object):

    def __init__(self, budget, auction_Items, lower_eval = 10,
            upper_eval= 150, p_irrationality=0.25):

        self.private_eval = dict()

        for it in auction_Items:
            self.private_eval[it.name] = np.random.uniform(lower_eval, upper_eval)

        self.irrationality = np.random.uniform(0, p_irrationality)
        self.budget = budget

        self.name = self._generate_id()

        self.chosen_items = self.roulette(self.private_eval.copy(), self.budget)

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
        """
        Should return 1 if bidder is ready to make bidder
        else return 0
        """


        if item_name not in self.chosen_items:
            return False



        if asking_price < self.private_eval[item_name]:
            ## TODO: The bidder should also try to manage his budget
            ## by prioritizing which item he whishes to buy
            return True

        if asking_price > self.private_eval[item_name]:
            #return False

            r = np.random.uniform()

            if r < self.irrationality:
                return True
            else:
                return False


    def __repr__(self):
        return repr((self.irrationality, self.budget))

    def _generate_id(self, n=6  , chars=ascii_uppercase):
        return "".join(random.choice(chars) for i in range(n))

class Auction_Items(object):

    def __init__(self):

        self.name       = self._generate_id()
        self.sold_to    = None     # indicates which biddeer bought the item
        self.solf_for   = None    # The price the item was sold for
        self.highest_eval = 0   # The maximal evaluation of this object by any bidder
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
                    still_bidding[0].budget -= asking_price
                    break


            asking_price += incremental_bid

def main(num_items, num_bidders, budget):

    auc_items = [Auction_Items() for _ in range(num_items)]

    bidders = [Bidder(budget, auc_items) for _ in range(num_bidders)]

    for it in auc_items:
        it.get_best_eval(bidders)

    english_auction(bidders, auc_items, 10, 0.5)

    for it in auc_items:
        print(it.name, it.sold_to.name, it.highest_eval, it.sold_for, it.second_highest_eval)

    for bidder in bidders:
        print(bidder.name)
        print(bidder.chosen_items)
        print(bidder.private_eval)

if __name__ == "__main__":
    num_bidders = 10
    num_items = 5
    budget = 250

    main(num_items, num_bidders, budget)
