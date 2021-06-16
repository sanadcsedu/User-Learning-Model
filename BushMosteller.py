import numpy as np
from collections import defaultdict
import pdb


class BushMosteller:
    def __init__(self, alpha, beta, state):
        self.alpha = alpha
        self.beta = beta
        self.prob = defaultdict(float)
        self.cond_prob = defaultdict(lambda: defaultdict(float))
        self.state = state

    # User already has some prior knowledge about some strategies regarding the given task
    def add_prior_strategies(self, priors):
        if self.state:
            for prev in priors:
                for cur in priors:
                    self.cond_prob[prev][cur] = (1 - self.cond_prob[prev][cur]) * self.alpha
        else:
            for attr in priors:
                self.prob[attr] = (1 - self.prob[attr]) * self.alpha
            # self.prob[attr] -= self.prob[attr] * self.beta

    def update(self, user, interactions, r):
        if not self.state:
            cur = interactions
            for attr in cur:
                if r > 0:
                    self.prob[attr] += (1 - self.prob[attr]) * self.alpha
                else:
                    self.prob[attr] -= self.prob[attr] * self.beta
            for keys in self.prob:
                if keys not in cur:
                    if r > 0:
                        self.prob[keys] -= self.prob[keys] * self.alpha
                    else:
                        self.prob[keys] += (1 - self.prob[keys]) * self.beta
        else:
            if len(interactions) < 2:
                return
            prev = interactions[0]
            cur = interactions[1]
            for prev_attr in prev:
                for cur_attr in cur:
                    if r > 0:
                        self.cond_prob[prev_attr][cur_attr] += (1 - self.cond_prob[prev_attr][cur_attr]) * self.alpha
                    else:
                        self.cond_prob[prev_attr][cur_attr] -= self.cond_prob[prev_attr][cur_attr] * self.beta

            for prev_attr in self.cond_prob:
                if prev_attr in prev:
                    continue
                for cur_attr in self.cond_prob[prev_attr]:
                    if cur_attr in cur:
                        continue
                    if r > 0:
                        self.cond_prob[prev_attr][cur_attr] -= self.cond_prob[prev_attr][cur_attr] * self.alpha
                    else:
                        self.cond_prob[prev_attr][cur_attr] += (1 - self.cond_prob[prev_attr][cur_attr]) * self.beta

    def normalize(self):
        if self.state:
            # pdb.set_trace()
            for prev_attr in self.cond_prob:
                sum = 0
                for cur_attr in self.cond_prob[prev_attr]:
                    sum += self.cond_prob[prev_attr][cur_attr]
                for cur_attr in self.cond_prob[prev_attr]:
                    # print("prev {} cur {}".format(prev_attr, cur_attr))
                    self.cond_prob[prev_attr][cur_attr] /= sum
            # print(self.cond_prob)
        else:
            sum = 0
            for attr in self.prob:
                sum += self.prob[attr]
            for attr in self.prob:
                self.prob[attr] /= sum

    #This method selects action with highest probablity
    # def select_from_ptable(self, temp_prob, k):
    #     ret_list = []
    #     total_prob = 0
    #     while k > 0:
    #         k -= 1
    #         cur_max = -1
    #         for attr in temp_prob:
    #             if cur_max < temp_prob[attr]:
    #                 cur_max = temp_prob[attr]
    #                 ret = []
    #                 ret.append(attr)
    #             elif cur_max == temp_prob[attr]:
    #                 ret.append(attr)
    #
    #         picked = np.random.randint(0, len(ret))
    #         ret_list.append(ret[picked])
    #         total_prob += cur_max
    #         del temp_prob[ret[picked]]
    #
    #     return ret_list

    #This method selects action proportional to probablity distribution
    def select_from_ptable(self, temp_prob, k):
        ret_list = []
        total_prob = 0
        tempk = k
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

        threshold = np.random.random()
        if threshold > total_prob:
            ret_list = self.random_selection(tempk)

        return ret_list

    def make_choice_state(self, user, interactions, k):
        if len(interactions) < 2:
            return self.random_selection(k)
        self.normalize()
        temp_cond_prob = self.cond_prob.copy()

        prev = interactions[1]
        temp_prob = defaultdict(float)

        norm = 0
        for all_strat in temp_cond_prob:
            prob = 1
            for prev_strat in prev:
                prob *= temp_cond_prob[prev_strat][all_strat]
            temp_prob[all_strat] = prob
            norm += prob

        for attrs in temp_prob:
            temp_prob[attrs] /= norm

        return self.select_from_ptable(temp_prob, k)

    def make_choice_nostate(self, k):
        self.normalize()
        prob = self.prob.copy()
        return self.select_from_ptable(prob, k)

    def random_selection(self, k):
        if self.state:
            choices = list(self.cond_prob.keys())
        else:
            choices = list(self.prob.keys())
        ret = []
        while k > 0:
            pick = np.random.randint(0, len(choices))
            ret.append(choices[pick])
            del choices[pick]
            k -= 1
        return ret
