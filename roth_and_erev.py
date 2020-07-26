import numpy as np
from collections import defaultdict


class roth_and_erev:

    def __init__(self):
        self.strategies = None
        self.cutoff = None
        # self.forgetting = None
        # self.attributes = attribute
        self.q_value = defaultdict(lambda: defaultdict())
        self.prob = defaultdict(lambda: defaultdict())

    #the Q values are updated using the following function and probablities are immediately calculated
    def update_qtable(self, user, attr, payoff, forgetting):
        if user in self.q_value:
            if attr in self.q_value[user]:
                self.q_value[user][attr] += payoff
            else:
                self.q_value[user][attr] = payoff
        else:
            self.q_value[user][attr] = payoff

        #Introducing the Forgetting parameter
        #see if numpy like operation can be done, for now:
        for attr in self.q_value[user]:
            self.q_value[user][attr] *= (1 - forgetting)

    #Get the probability of each strategy from the Q values
    def update_prob_qtable(self, user):
        # summ = np.sum(self.q_value[user].values())
        sum = 0
        for attr in self.q_value[user]:
            sum += self.q_value[user][attr]

        for attr in self.q_value[user]:
            self.prob[user][attr] = self.q_value[user][attr] / sum

    # Just for checking if everything is working properly
    def tester(self, user):
        print(self.q_value[user])
        # print(self.prob[user].values())
        # test = 0
        # for attr in self.prob[user]:
        #     test += self.prob[user][attr]
        # print(test)

    #Picking up a strategy based on the Q-values. If the algorithm is not confident than a strategy
    # is randomly picked
    def make_choice(self, user, k, threshold, attributes):
        # print(k)
        ret_attr = defaultdict()
        temp_attr_list = attributes
        temp_prob = self.prob[user]
        # print(temp_prob)
        # print(temp_attr_list.keys())
        while k >= 0:
            ret = None
            cur_max = -1
            for attr in temp_prob:
                if cur_max < temp_prob[attr]:
                    cur_max = temp_prob[attr]
                    ret = attr

            n = len(temp_attr_list)
            if cur_max < threshold:
                pick = np.random.randint(0, n)
                cnt = 0
                for attr in temp_attr_list:
                    if cnt == pick:
                        ret_attr[attr] = 1
                        del temp_attr_list[attr]
                        if attr in temp_prob:
                            del temp_prob[attr]
                        break
                    cnt += 1
            else:
                ret_attr[ret] = 1
                del temp_prob[ret]
                if ret in temp_attr_list:
                    del temp_attr_list[ret]
            k -= 1
        return ret_attr

    #Choosing strategy.(k strategies needs to be choosen to calculate P@k)
    def make_choice_v2(self, user, k, threshold):
        self.update_prob_qtable(user)
        temp_prob = self.prob[user]

        # print(len(temp_prob))
        # print(temp_prob)
        # print(temp_prob.keys())

        ret = None
        ret_list = []
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
        return ret_list

    def random_choice(self, user, k):
        ret_list = []
        temp_attr_list = list(self.q_value[user].keys())
        # print(temp_attr_list)
        while k > 0:
            n = len(temp_attr_list)
            # print("N = {}".format(n))
            pick = np.random.randint(0, n)
            # print(pick)
            ret_list.append(temp_attr_list[pick])
            # print(temp_attr_list[pick])
            del temp_attr_list[pick]
            k -= 1

        return ret_list
