from _collections import defaultdict
import numpy as np
import pdb
import queue

#Has the Win-Keep Lose-Randomize algorithm
#Also contains completely Randomize algorithm that reflects an irrational

class WinKeepLoseRandomize:
    def __init__(self, all_attrs, final):
        self.strategies = defaultdict(list, {key: 0 for key in all_attrs})
        self.rewarded_stratgies = final
        self.win = False #Tracks if the last action was a win / lose
        self.keep = list() #Tracks the last strategy

#K denotes how many attributes should be returned
    def make_choice(self, k):
        # if self.win:
        #     return self.keep
        # else:
        self.keep = self.randomized_choice(k)
        # self.win = False
        return self.keep

    def assign_reward(self, picked_attrs):
        # flag = False
        for attr in picked_attrs:
            if attr not in self.rewarded_stratgies:
                self.keep.remove(attr)
                # flag = True
                # break
        # pdb.set_trace()
        # self.win = flag


    def randomized_choice(self, k):
        choices = list(self.strategies.keys())
        ret = []
        for attrs in self.keep:
            ret.append(attrs)
            choices.remove(attrs)
        k -= len(self.keep)
        while k > 0:
            k -= 1
            pick = np.random.randint(len(choices))
            ret.append(choices[pick])
            choices.remove(choices[pick])
        # pdb.set_trace()
        return ret

#For testing purpose
if __name__ == "__main__":

    all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"',
                 '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"',
                 '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"',
                 '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"',
                 '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"',
                 '"tornado"']
    obj = WinKeepLoseRandomize(all_attrs)
    r = obj.randomized_choice(5)
    print(r)