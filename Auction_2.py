import numpy as np
import random
from string import ascii_uppercase
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
from matplotlib import cm
import argparse

'''
    # TODO:
    Add Dutch auction
    Add multiple round logic
    Change object for each round ??
'''
#######################------Argumnet Parser-------------######################
parser = argparse.ArgumentParser()
parser.add_argument("-A", "--auctionType", help = "Type of Auction either dutch or English", default="Englih")

args = parser.parse_args()
auction_type = args.auctionType

######################---------Argumnet Parser----------#######################

#######################-------Global Variabls-------------#####################
num_bidders = 100
num_items = 5
budget = 200
num_iter = 100
init_bid = 10
incremental_bid = 1
#####################---------Global Variabls----------------###################

class Bidder(object):

    def __init__(self, budget, p_irrationality=0.5):

        self.irrationality = np.random.uniform(-p_irrationality, p_irrationality)
        self.budget = budget
        self.name = self._generate_id()
        self.num_items_bought = 0
        self.history = []

        self.prio_bool = random.choice([True, False])

    def roulette(self, eval_items, available_budget):

        chosen_item = []

        if self.prio_bool == False:
            return list(eval_items.keys())


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

        if self.budget < 0:
            return False
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
        bi = bidders.copy()
        bi.sort(key=lambda x : x.private_eval[self.name], reverse=True)

        self.highest_eval = bi[0].private_eval[self.name]
        self.second_highest_eval = bi[1].private_eval[self.name]

def english_auction(bidders, Auction_Items, initial_bid, incremental_bid):

    for it in Auction_Items:
        still_bidding = bidders.copy()
        asking_price = initial_bid
        while not it.sold_to:

            for bidder in still_bidding:
                bid = bidder.make_bid(it.name, asking_price)

                if bid == False:
                    still_bidding.remove(bidder)

                if len(still_bidding) == 1:
                    it.sold_to  = still_bidding[0]
                    it.sold_for = asking_price

                    payoff = still_bidding[0].private_eval[it.name] - asking_price
                    still_bidding[0].budget += payoff

                    still_bidding[0].num_items_bought += 1
                    break


            asking_price += incremental_bid


def dutch_auction(bidders, Auction_items, initial_bid, inc_bid):

    #Iterate over each item
    for it in Auction_items:

        asking_price = initial_bid

        #As long the item is not sold
        while not it.sold_to:

            for bidder in bidders:
                bid = bidder.make_bid(it.name, asking_price)

                if bid == True:
                    it.sold_to = bidder
                    it.sold_for = asking_price

                    payoff = bidder.private_eval[it.name] - asking_price
                    bidder.budget += payoff

                    bidder.num_items_bought += 1

            asking_price -= inc_bid


def main(auction_type = 'English'):

    #Generate new items for auction
    auc_items = [Auction_Items() for _ in range(num_items)]

    #For Each bidder set privte eval
    for bidder in bidders:
        bidder.set_private_eval(auc_items)

    #For each item find best evaluation
    for it in auc_items:
        it.get_best_eval(bidders)

    #For each bidder generate a list of items to bid on
    for bidder in bidders:
        bidder.chosen_item = bidder.roulette(bidder.private_eval.copy(), bidder.budget)

    #Play english auction
    if auction_type == "English":
        english_auction(bidders, auc_items, init_bid, incremental_bid)

    elif auction_type == 'Dutch':
        dutch_auction(bidders, auc_items, 1000, incremental_bid)

    for bidder in bidders:
        bidder.history.append(bidder.budget)


##Generate a  fix set of bidders
bidders = [Bidder(budget) for _ in range(num_bidders)]


###################set up the plot###################
viridis = cm.get_cmap('gist_rainbow', 40)
colours = viridis(np.linspace(0,1,num_bidders))

line_styes = ['solid', 'dashed', 'dashdot', 'dotted']

fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(0, 400))


lines = []

for ic, bidder in zip(colours, bidders):

    lb = "Prio={}, irr={:.2f}".format(bidder.prio_bool, bidder.irrationality)
    line, = ax.plot([], [], lw = 2, label=lb, color=ic, ls = random.choice(line_styes))
    lines.append(line)


plt.legend(loc='upper left', ncol=5)
plt.title(auction_type + " Auction")
plt.xlabel("Itterations")
plt.ylabel("Budget")



#########---------------------Set up the plot-------------------------########

def init():

    for line in lines:
        line.set_data([],[])
    return lines

def animate(i):

    main(auction_type)
    n = len(bidders[0].history)
    x = list(range(0, n))

    max_y = 0


    for line, bidder in zip(lines, bidders):


        line.set_data(x, bidder.history)

        if bidder.budget > max_y:
            max_y = bidder.budget


    if max_y > ax.get_ylim()[1]:
        ax.set_ylim(0, max_y + 250)
    if i > ax.get_xlim()[1]:
        ax.set_xlim(0, i + 100)




    return lines

ani = FuncAnimation(fig, animate, frames= 300,
                    interval = 100, init_func=init, repeat=False)



plt.show()


dt = np.zeros((num_bidders, 3))

fig3, ax3 = plt.subplots()

for idx, bidder in enumerate(bidders):
    dt[idx,0] = bidder.history[-1]
    dt[idx,1] = bidder.irrationality
    dt[idx,2] = bidder.prio_bool

cdict = {False: "red", True: "blue"}

for g in np.unique(dt[:,2]):
    lb = "No prio" if g == False else "prio"
    ix = np.where(dt == g)
    ax3.scatter(dt[ix,1], dt[ix,0], c = cdict[g], label = lb, s = 25)

    a = dt[ix[0],:]
    a = a[a[:,1].argsort()]
    ax3.plot(a[:,1], a[:,0], color=cdict[g], ls = 'dashed')
#dt = dt[dt[:,1].argsort()]
#ax3.plot(dt[:,1], dt[:,0], color='black', ls='dashed')
ax3.legend()
plt.title(auction_type + " Auction")
plt.ylabel("Budget")
plt.xlabel("irrationality factor")
plt.show()
