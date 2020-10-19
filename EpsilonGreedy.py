import pdb
import numpy as np
from _collections import defaultdict

class EpsilonGreedy:
    def __init__(self, all_attrs, epsilon):
        self.strategies = defaultdict(list,{key: 0 for key in all_attrs})
        self.cond_strat = defaultdict(lambda: defaultdict(float))
        self.q = defaultdict(list,{key: 0 for key in all_attrs})
        self.c_q = defaultdict(lambda : defaultdict(float))
        self.epsilon = epsilon

    def make_choice(self, k, interactions):
        ret = []
        if np.random.random() > self.epsilon:
            #return K strategies with maximum probabilities
            ret = self.best_k_selection(interactions, k)
        else:
            #return k strategies Randomly
            ret = self.random_selection(k)
        return ret

    def best_k_selection(self, interactions, k):
        if len(interactions) < 2:
            return self.random_selection(k)
        # print("Best")
        ret_list = defaultdict()
        prob = self.q.copy()
        cond_prob = self.c_q.copy()
        prev = interactions[1]
        while k > 0:
            ret = []
            _max = -1
            for attr in prob:
                if _max < prob[attr]:
                    _max = prob[attr]
                    ret = []
                    ret.append(attr)
                elif _max == prob[attr]:
                    ret.append(attr)

            for prev_attr in prev:
                for cur_attr in cond_prob[prev_attr]:
                    if _max < cond_prob[prev_attr][cur_attr]:
                        _max = cond_prob[prev_attr][cur_attr]
                        ret = []
                        ret.append(cur_attr)
                    elif _max == cond_prob[prev_attr][cur_attr]:
                        ret.append(cur_attr)

            picked = np.random.randint(0, len(ret))
            ret_list[ret[picked]] = 1

            del prob[ret[picked]]
            for prev_attr in prev:
                if ret[picked] in cond_prob[prev_attr]:
                    del cond_prob[prev_attr][ret[picked]]

            k -= 1

        # pdb.set_trace()
        return list(ret_list.keys())

    def random_selection(self, k):
        # print("random")
        choices = list(self.strategies.keys())
        ret = []
        while k > 0:
            pick = np.random.randint(0, len(choices))
            ret.append(choices[pick])
            del choices[pick]
            k -= 1
        return ret

    def update(self, interactions, reward):
        sz = len(interactions)
        if sz == 0:
            return

        elif sz == 1:
            cur = interactions[sz - 1]
            for attr in cur:
                self.strategies[attr] += 1
                n = self.strategies[attr]
                self.q[attr] += self.q[attr] * ((n - 1) / n) + (reward / n)

        else:
            prev = interactions[0]
            cur = interactions[1]
            for prev_attr in prev:
                for cur_attr in cur:
                    self.cond_strat[prev_attr][cur_attr] += 1
                    n = self.cond_strat[prev_attr][cur_attr]
                    self.c_q[prev_attr][cur_attr] += self.c_q[prev_attr][cur_attr] * ((n - 1) / n) + (reward / n)
