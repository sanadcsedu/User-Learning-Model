import pdb
import numpy as np
from _collections import defaultdict
import math

#Includes both classical E-Greedy and Adaptive E-Greedy algorithm
class no_exploration:
    def __init__(self, all_attrs, epsilon, l, f, state):
        self.strategies = defaultdict(list,{key: 0 for key in all_attrs})
        self.cond_strat = defaultdict(lambda: defaultdict(float))
        self.q = defaultdict(list,{key: 0 for key in all_attrs})
        self.c_q = defaultdict(lambda : defaultdict(float))
        self.epsilon = epsilon
        self.t = 0 #Keeps track of how many times the algorithm performs exploration
        self.l = l #maximum number of time exploration can continue
        self.f = f #Regularization Parameter
        self.max_prev = 0
        self.max_cur = 0
        self.cnt = 0
        self.state = state

        for prev in all_attrs:
            for cur in all_attrs:
                self.c_q[prev][cur] = 1

    def make_choice_classic(self, k, interactions):
        ret = []
        ##############NO-EXPLORATION$$$$$$$$$$$$$$$$$$$$$$$$
        if self.state:
            ret = self.best_k_selection_state(interactions, k)
        else:
            ret = self.best_k_selection_nostate(k)
        ##############NO-EXPLORATION$$$$$$$$$$$$$$$$$$$$$$$$
        return ret

    def update(self, interactions, reward):
        self.max_cur += reward
        self.cnt += 1
        sz = len(interactions)
        if self.state:
            if sz < 2:
                return
            prev = interactions[0]
            cur = interactions[1]
            for prev_attr in prev:
                for cur_attr in cur:
                    self.cond_strat[prev_attr][cur_attr] += 1
                    n = self.cond_strat[prev_attr][cur_attr]
                    self.c_q[prev_attr][cur_attr] += self.c_q[prev_attr][cur_attr] * ((n - 1) / n) + (reward / n)
        else:
            cur = interactions[sz - 1]
            for attr in cur:
                self.strategies[attr] += 1
                n = self.strategies[attr]
                self.q[attr] += self.q[attr] * ((n - 1) / n) + (reward / n)

    def select_from_ptable(self, temp_prob, k):
        ret_list = []
        total_prob = 0
        while k > 0:
            k -= 1
            cur_max = -1
            for attr in temp_prob:
                if cur_max < temp_prob[attr]:
                    cur_max = temp_prob[attr]
                    ret = []
                    ret.append(attr)
                elif cur_max == temp_prob[attr]:
                    ret.append(attr)

            picked = np.random.randint(0, len(ret))
            ret_list.append(ret[picked])
            total_prob += cur_max
            del temp_prob[ret[picked]]

        return ret_list

    def best_k_selection_nostate(self, k):
        prob = self.q.copy()
        return self.select_from_ptable(prob, k)
