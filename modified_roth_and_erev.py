import numpy as np
from collections import defaultdict
from collections import Counter
import pdb

class modified_roth_and_erev:

    def __init__(self):
        self.cutoff = None
        # self.attributes = attribute
        self.q_value = defaultdict(lambda: defaultdict(float))
        self.cond_q_value = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.prob = defaultdict(lambda: defaultdict(float))
        self.cond_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    #User already has some prior knowledge about some strategies regarding the given task
    def add_prior_strategies(self, user, priors, payoff):
        for attr in priors:
            if user in self.q_value:
                if attr in self.q_value[user]:
                    self.q_value[user][attr] += payoff
                else:
                    self.q_value[user][attr] = payoff
            else:
                self.q_value[user][attr] = payoff

    #the Q values are updated using the following function and probablities are immediately calculated
    def update_qtable(self, user, interactions, payoff, forgetting):
        sz = len(interactions)
        if sz == 0:
            return
            # print(interactions)

        cur = interactions[sz - 1]
        # print(cur)

        #Updating Q-values for Pure Strategies similar to Basic Roth and Erev
        for attr in cur:
            if user in self.q_value:
                if attr in self.q_value[user]:
                    self.q_value[user][attr] += payoff
                else:
                    self.q_value[user][attr] = payoff
            else:
                self.q_value[user][attr] = payoff

            #Introducing the Forgetting parameter
            #see if numpy like operation can be done, for now:
            for strategies in self.q_value[user]:
                self.q_value[user][strategies] *= (1 - forgetting)

        #Updating Conditional Q-values
        if sz == 2:
            prev = interactions[0]
            for prev_attr in prev:
                for cur_attr in cur:
                    try:
                        self.cond_q_value[user][prev_attr][cur_attr] += payoff
                    except KeyError:
                        print("error: {} {}".format(prev_attr, cur_attr))

            #Introducing the Forgetting parameter
            #see if numpy like operation can be done, for now:
            for prev_strategies in self.cond_q_value[user]:
                for cur_strategies in self.cond_q_value[user][prev_strategies]:
                    self.cond_q_value[user][prev_strategies][cur_strategies] *= (1 - forgetting)

    #Get the probability of each strategy from the Q values
    def update_prob_qtable(self, user):
        # summ = np.sum(self.q_value[user].values())
        sum = 0
        for attr in self.q_value[user]:
            sum += self.q_value[user][attr]

        for attr in self.q_value[user]:
            self.prob[user][attr] = self.q_value[user][attr] / sum

        # sum = 0
        # for prev_attr in self.cond_q_value[user]:
        #     for cur_attr in self.cond_q_value[user][prev_attr]:
        #         sum += self.cond_q_value[user][prev_attr][cur_attr]
        #
        # for prev_attr in self.cond_q_value[user]:
        #     for cur_attr in self.cond_q_value[user][prev_attr]:
        #         self.cond_prob[user][prev_attr][cur_attr] = self.cond_q_value[user][prev_attr][cur_attr] / sum

        for prev_attr in self.cond_q_value[user]:
            sum = 0
            for cur_attr in self.cond_q_value[user][prev_attr]:
                sum += self.cond_q_value[user][prev_attr][cur_attr]

            for cur_attr in self.cond_q_value[user][prev_attr]:
                self.cond_prob[user][prev_attr][cur_attr] = self.cond_q_value[user][prev_attr][cur_attr] / sum

    # Just for checking if everything is working properly
    def tester(self, user):
        # print(self.q_value[user])
        self.update_prob_qtable(user)
        print("Conditional Dependencies")
        print(self.cond_prob[user])
        # print(self.prob[user].values())
        # test = 0
        # for attr in self.prob[user]:
        #     test += self.prob[user][attr]
        # print(test)

    #Choosing strategy.(k strategies needs to be choosen to calculate P@k)
    def make_choice(self, user, prev_inter, k, threshold):
        self.update_prob_qtable(user)
        temp_prob = self.prob[user]
        temp_cond_prob = self.cond_prob[user]

        # print(len(temp_prob))
        # print(temp_prob)
        # print(temp_prob.keys())
        sz = len(prev_inter)
        ret = None
        ret_list = []

        if sz < 1:
            while k > 0:
                k -= 1
                cur_max = -1
                for attr in temp_prob:
                    if cur_max < temp_prob[attr]:
                        cur_max = temp_prob[attr]
                        ret = attr

                if cur_max < threshold:
                    cnt = 0
                    pick = np.random.randint(0, len(temp_prob))
                    for attr in temp_prob:
                        if cnt == pick:
                            ret = attr
                            break
                        cnt += 1

                del temp_prob[ret]
                ret_list.append(ret)
        else:
            prev = prev_inter[sz - 1]
            choose_from = defaultdict()
            for attr in prev:
                if choose_from is None:
                    choose_from = temp_cond_prob[attr]
                else:
                    choose_from.update(temp_cond_prob[attr])
            # print(choose_from)
            while k > 0:
                if len(choose_from) == 0:
                    # print("Choose from failed {}".format(prev))
                    choose_from = temp_prob
                k -= 1
                cur_max = -1
                for attr in choose_from:
                    if cur_max < choose_from[attr]:
                        cur_max = choose_from[attr]
                        ret = attr

                if cur_max < threshold:
                    cnt = 0
                    try:
                        pick = np.random.randint(0, len(choose_from))
                    except:
                        # pdb.set_trace()
                        break
                    # print("Pick {} Length {}".format(pick, len(choose_from)))
                    for attr in choose_from:
                        if cnt == pick:
                            ret = attr
                            break
                        cnt += 1

                del choose_from[ret]
                ret_list.append(ret)
            # print("Choose from: {}".format(prev), end=" ")
            # print(choose_from)

        return ret_list