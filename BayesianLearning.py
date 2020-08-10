from collections import defaultdict
import numpy as np
import pdb

class BayesianLearning:
    def __init__(self, hypothesis):
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.posterior = defaultdict(lambda: defaultdict(float))
        self.prior = defaultdict(float)
        self.hypo = hypothesis
        # print(hypothesis)
        # print(len(hypothesis))

    def set_prior(self, hypothesis):
        sz = len(hypothesis)
        for attr in hypothesis:
            self.prior[attr] = 1/sz
        # print(self.prior)

    def update_prior(self):
        sum = 0
        for attr in self.prior:
            sum += self.prior[attr]

        for attr in self.prior:
            self.prior[attr] /= sum


    #Updating the likelihood function P(d | h_i)
    def update_likelihood(self, interactions, UpdatePrior = False):
        d = interactions[0]  # Previous Interactions / hypothesis (attributes)
        h_i = interactions[1] #Current Interactions / observed value
        for prev_attr in d:
            for cur_attr in h_i:
                if UpdatePrior:
                    self.prior[cur_attr] += 1
                self.likelihood[prev_attr][cur_attr] += 1

        for observed in self.likelihood:
            sum = 0
            for hypothesis in self.likelihood[observed]:
                sum += self.likelihood[observed][hypothesis]

            if sum == 0:
                continue

            for hypothesis in self.likelihood[observed]:
                self.likelihood[observed][hypothesis] /= sum

    #Updating the Posterior function P(h_i | d) = alpha * P(d | h_i) * P(h_i)
    def get_posterior(self, interactions, k):
        sz = len(interactions)
        ret = None
        ret_list = []

        hypothesis_set = self.hypo.copy()
        # print(type(hypothesis_set))
        #As no previous interaction, choice is made randomly because of Uniform Prior.
        if sz < 1:
            while k > 0:
                k -= 1
                pick = np.random.randint(0, len(hypothesis_set))
                cnt = 0
                for attr in hypothesis_set:
                    if cnt == pick:
                        ret = attr
                        break
                    cnt += 1

                # del hypothesis_set[ret]
                hypothesis_set.remove(ret)
                ret_list.append(ret)
        else:
            prev = interactions[sz - 1]
            while k > 0:
                # try:
                for observed in prev:
                    max_prob = 0
                    for choice in hypothesis_set:
                        if max_prob < self.likelihood[observed][choice] * self.prior[choice]:
                            max_prob = self.likelihood[observed][choice] * self.prior[choice]
                            ret = choice

                    if max_prob == 0:
                        pick = np.random.randint(0, len(hypothesis_set))
                        cnt = 0
                        for attr in hypothesis_set:
                            if cnt == pick:
                                ret = attr
                                break
                            cnt += 1

                    # print(ret)
                    # del hypothesis_set[ret]
                    hypothesis_set.remove(ret)
                    ret_list.append(ret)
                    k -= 1
                    if k == 0:
                        return ret_list

        return ret_list



