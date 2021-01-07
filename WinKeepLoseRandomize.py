from _collections import defaultdict
import numpy as np
import pdb
import queue

#Has the Win-Keep Lose-Randomize algorithm [https://en.wikipedia.org/wiki/Win%E2%80%93stay,_lose%E2%80%93switch]
#Also contains completely Randomize algorithm that reflects an irrational

class WinKeepLoseRandomize:
    def __init__(self, all_attrs, final):
        self.strategies = defaultdict(list, {key: 0 for key in all_attrs})
        self.rewarded_stratgies = final
        # print(self.rewarded_stratgies)
        self.win = False #Tracks if the last action set resulted in a win / lose
        self.keep = list() #Records the last used action set

    #K denotes how many attributes should be returned
    def make_choice(self, k):
        if self.win:
            return self.keep #If the previous action set was useful use the same set of actions
        else:
            self.keep = self.randomized_choice(k) # previous interaction was a failure, randomly select actions
            # print("Random {}".format(self.keep))
        return self.keep

    def assign_reward(self, picked_attrs):
        flag = False
        for attr in picked_attrs:
            if attr in self.rewarded_stratgies:
                self.keep = picked_attrs
                flag = True
                break
        self.win = flag
        # pdb.set_trace()


    def randomized_choice(self, k):
        choices = list(self.strategies.keys())
        ret = []
        while k > 0:
            k -= 1
            pick = np.random.randint(len(choices))
            ret.append(choices[pick])
            choices.remove(choices[pick])
        return ret