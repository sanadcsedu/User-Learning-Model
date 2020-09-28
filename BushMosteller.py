import numpy as np
from collections import defaultdict

class BushMosteller:
    def __init__(self, alpha, strategies):
        self.alpha = alpha
        self.prob = defaultdict(float)
        self.cond_prob = defaultdict(lambda: defaultdict(float))

    def update(self, interactions):
        sz = len(interactions)
        if sz == 0:
            return

        elif sz == 1:
            cur = interactions[sz - 1]
            for attr in cur:
                self.prob[attr] += (1 - self.prob[attr]) * self.alpha
            for keys in self.prob:
                if keys not in cur:
                    self.prob[attr] -= self.prob[attr] * self.alpha

        else:
            prev = interactions[0]
            cur = interactions[1]
            for prev_attr in prev:
                for cur_attr in cur:
                    self.cond_prob[prev_attr][cur_attr] += (1 - self.cond_prob[prev_attr][cur_attr]) * self.alpha

            for prev_attr in self.cond_prob:
                if prev_attr in prev:
                    continue
                for cur_attr in self.cond_prob[prev_attr]:
                    if cur_attr in cur:
                        continue
                    self.cond_prob[prev_attr][cur_attr] -= self.cond_prob[prev_attr][cur_attr] * self.alpha

    def normalize(self):
        sum = 0
        for attr in self.prob:
            sum += self.prob[attr]
        for prev_attr in self.cond_prob:
            for cur_attr in self.cond_prob[prev_attr]:
                sum += self.cond_prob[prev_attr][cur_attr]

        for attr in self.prob:
            self.prob[attr] /= sum
        for prev_attr in self.cond_prob:
            for cur_attr in self.cond_prob[prev_attr]:
                self.cond_prob[prev_attr][cur_attr] /= sum

    def make_choice(self, k):
        self.normalize()
        ret_list = defaultdict()
        ret = None
        while k > 0:
            _max = -1
            for attr in self.prob:
                if _max < self.prob[attr] and attr not in ret_list:
                    _max = self.prob[attr]
                    ret = attr
            for prev_attr in self.cond_prob:
                for cur_attr in self.cond_prob[prev_attr]:
                    if _max < self.cond_prob[prev_attr][cur_attr] and cur_attr not in ret_list:
                        _max = self.cond_prob[prev_attr][cur_attr]
                        ret = cur_attr
            ret_list[ret] = 1
            k -= 1

        return list(ret_list.keys())





