import numpy as np
from collections import defaultdict
from EpsilonGreedy import EpsilonGreedy
import queue
import Categorizing
import Evaluators
import pdb
import read_data

class TrainGreedy:

    def __init__(self):
        self.users = None
        self.all_attrs = None
        self.priors = None
        self.final = None

    def set_data(self, users, all_attrs, priors, final):
        self.users = users
        self.all_attrs = all_attrs
        self.priors = priors
        self.final = final

    def run_epsilon_greedy(self, cur_data, user, dataset, cur_task, k, epsilon, state, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        total = 0
        after = 2
        epg = EpsilonGreedy(self.all_attrs, epsilon, 0, 0, state)

        f1_score = 0
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)

            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = epg.make_choice_classic(k, list(prev_interactions.queue))
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)

                if cat:
                    ground = c.get_category(ground)

                if len(ground) == 0:
                    continue

                if prev_interactions.full():
                    prev_interactions.get()
                    prev_interactions.put(ground)
                else:
                    prev_interactions.put(ground)

                #Payoff is calculated based on the number of *correct attributes* in the current interaction
                payoff = 0
                for attrs in picked_attr:
                    if attrs in self.final:
                        payoff += 1

                if payoff > 0:
                    epg.update(list(prev_interactions.queue), payoff)

                if no_of_intr >= after:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score += get_f1
                    total += 1
                no_of_intr += 1

        f1_score = f1_score / total
        return f1_score

    def run_adaptive_epsilon_greedy(self, cur_data, user, dataset, cur_task, k, epsilon, l, f, state, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        total = 0
        after = 2
        aepg = EpsilonGreedy(self.all_attrs, epsilon, l, f, state)

        f1_score = 0
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)

            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = aepg.make_choice_adaptive(k, list(prev_interactions.queue))
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)

                if cat:
                    ground = c.get_category(ground)
                # pdb.set_trace()
                if len(ground) == 0:
                    continue

                if prev_interactions.full():
                    prev_interactions.get()
                    prev_interactions.put(ground)
                else:
                    prev_interactions.put(ground)

                payoff = 0
                for attrs in picked_attr:
                    if attrs in final:
                        payoff += 1

                if payoff > 0:
                    aepg.update(list(prev_interactions.queue), payoff)

                if no_of_intr >= after:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score += get_f1
                    total += 1
                no_of_intr += 1

        f1_score = f1_score / total
        return f1_score


if __name__ == '__main__':
    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = 'faa1'
    task = 't1'
    users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
    greedy = TrainGreedy()
    greedy.set_data(users, all_attrs, priors, final)

    f1_score = 0
    epoch = 10
    k = 3

    # epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    states = [False, True]
    for state in states:
    # for epsilon in epsilons:
        print(state)
        epsilon = 0.05
        f1_score = 0
        for user in greedy.users:
            avrg_user = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                accu = greedy.run_epsilon_greedy(data, user, dataset, task, k, epsilon, state, True)
                # print("User: {} Precision@K {}".format(user, accu))
                avrg_user += accu
                # break
            f1_score += (avrg_user / epoch)
            # break
        f1_score /= len(greedy.users)
        print("F1 Score Epsilon {} Greedy (No State) = {}".format(epsilon, f1_score))

        f1_score = 0
        epoch = 10
        # k = 4
        epsilon = 0.25
        # state = True
        # ls = [2, 3, 4, 5]
        # fs = [1, 2, 3, 4, 5]
        l = 4
        f = 3
        for user in greedy.users:
            avrg_user = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                accu = greedy.run_adaptive_epsilon_greedy(data, user, dataset, task, k, epsilon, l, f, state, True)
                # print("User: {} Precision@K {}".format(user, accu))
                avrg_user += accu
                # break
            f1_score += (avrg_user / epoch)
            # break
        f1_score /= len(greedy.users)
        print("l {} f {} F1 Score Adaptive Epsilon Greedy (No State) = {}".format(l, f, f1_score))
    #


