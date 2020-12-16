import numpy as np
from collections import defaultdict
from collections import Counter
import pdb

class modified_roth_and_erev:

    def __init__(self, state):
        self.cutoff = None
        self.state = state
        self.q_value = defaultdict(lambda: defaultdict(float))
        self.cond_q_value = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.prob = defaultdict(lambda: defaultdict(float))
        self.cond_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    #User already has some prior knowledge about some strategies regarding the given task
    def add_prior_strategies(self, user, priors, payoff):
        if self.state:
            for prev in priors:
                for cur in priors:
                    self.cond_q_value[user][prev][cur] = payoff
        else:
            for attr in priors:
                self.q_value[user][attr] += payoff


    #the Q values are updated using the following function and probablities are immediately calculated
    def update_qtable(self, user, interactions, payoff, forgetting):
        if self.state:
            if len(interactions) < 2:
                return
            prev = interactions[0]
            cur = interactions[1]
            for prev_attrs in prev:
                for cur_attr in cur:
                    self.cond_q_value[user][prev_attrs][cur_attr] += payoff

            for prev_strat in self.cond_q_value[user]:
                for cur_strat in self.cond_q_value[user][prev_strat]:
                    self.cond_q_value[user][prev_strat][cur_strat] *= (1 - forgetting)
        else:
            #Updating Q-values for Pure Strategies similar to Basic Roth and Erev
            for attr in interactions:
                self.q_value[user][attr] += payoff

            #Introducing the Forgetting parameter
            for strategies in self.q_value[user]:
                self.q_value[user][strategies] *= (1 - forgetting)


    #Get the probability of each strategy from the Q values
    def update_prob_qtable(self, user):
        if self.state:
            for prev_attr in self.cond_q_value[user]:
                sum = 0
                for cur_attr in self.cond_q_value[user][prev_attr]:
                    sum += self.cond_q_value[user][prev_attr][cur_attr]

                for cur_attr in self.cond_q_value[user][prev_attr]:
                    self.cond_prob[user][prev_attr][cur_attr] = self.cond_q_value[user][prev_attr][cur_attr] / sum
        else:
            sum = 0
            for attr in self.q_value[user]:
                sum += self.q_value[user][attr]
            #Normalizing
            for attr in self.q_value[user]:
                self.prob[user][attr] = self.q_value[user][attr] / sum

    def select_from_ptable(self, user, temp_prob, k):
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
            ret_list = self.random_selection(user, tempk)

        return ret_list

    def random_selection(self,user, k):
        if self.state:
            choices = list(self.cond_q_value[user].keys())
        else:
            choices = list(self.prob[user].keys())
        ret = []
        while k > 0:
            pick = np.random.randint(0, len(choices))
            ret.append(choices[pick])
            del choices[pick]
            k -= 1
        return ret

    def make_choice_nostate(self, user, k):
        self.update_prob_qtable(user)
        temp_prob = self.prob[user].copy()
        return self.select_from_ptable(user, temp_prob, k)

    def make_choice_state(self, user, prev_inter, k):
        self.update_prob_qtable(user)

        if len(prev_inter) < 2:
            # print(user)
            return self.random_selection(user, k)

        temp_cond_prob = self.cond_prob[user].copy()

        prev = prev_inter[1]
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
        # print(temp_prob)
        return self.select_from_ptable(user, temp_prob, k)

