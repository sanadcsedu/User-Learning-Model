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
        self.threshold = 0.4

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

        f1_score = []
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
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
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1

        return e.before_after(f1_score, total, self.threshold)

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

        f1_score = []
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
                if len(ground) == 0:
                    continue

                if prev_interactions.full():
                    prev_interactions.get()
                    prev_interactions.put(ground)
                else:
                    prev_interactions.put(ground)

                payoff = 0
                for attrs in picked_attr:
                    if attrs in self.final:
                        payoff += 1

                if payoff > 0:
                    aepg.update(list(prev_interactions.queue), payoff)

                if no_of_intr >= after:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1

        return e.before_after(f1_score, total, self.threshold)

    def hyperparameter_classic(self):
        obj = read_data.read_data()
        obj.create_connection(r"D:\Tableau Learning\Tableau.db")
        dataset = 'birdstrikes1'
        task = 't4'
        users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
        self.set_data(users, all_attrs, priors, final)

        epoch = 10
        k = 3
        epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        states = [False]
        _max = best_epsilon = -1
        for state in states:
            for epsilon in epsilons:
                f1_score = 0
                for user in self.users:
                    avrg_user = 0
                    data = obj.read_cur_data(user, dataset)
                    for experiment in range(epoch):
                        accu_b, accu_a, accu = self.run_epsilon_greedy(data, user, dataset, task, k, epsilon, state, True)
                        avrg_user += accu
                    f1_score += (avrg_user / epoch)
                f1_score /= len(self.users)
                if _max < f1_score:
                    _max = f1_score
                    best_epsilon = epsilon

        print(" Epsilon {} Greedy (no state) -> F1 Score= {}".format(best_epsilon, _max))

    def hyperparameter_adaptive(self):
        obj = read_data.read_data()
        obj.create_connection(r"D:\Tableau Learning\Tableau.db")
        dataset = 'birdstrikes1'
        task = 't4'
        users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
        self.set_data(users, all_attrs, priors, final)

        epoch = 10
        k = 3
        state = False
        epsilon = 0.05
        ls = [2, 3, 4, 5, 6, 7, 8, 9, 10] #Maximum number of time exploration can continue
        fs = [1, 2, 3, 4, 5] #Regularization Parameter
        f1_score = 0
        best_f = best_l = _max = -1
        for l in ls:
            for f in fs:
                for user in self.users:
                    avrg_user = 0
                    data = obj.read_cur_data(user, dataset)
                    for experiment in range(epoch):
                        accu_b, accu_a, accu = self.run_adaptive_epsilon_greedy(data, user, dataset, task, k, epsilon, l, f, state, True)
                        avrg_user += accu
                    f1_score += (avrg_user / epoch)
                f1_score /= len(self.users)
                if _max < f1_score:
                    _max = f1_score
                    best_f = f
                    best_l = l
        print("l {} f {} F1 Score Adaptive Epsilon Greedy (no-state) = {}".format(best_l, best_f, _max))


if __name__ == '__main__':
    greedy = TrainGreedy()
    greedy.hyperparameter_classic()
    # greedy.hyperparameter_adaptive()

    # obj = read_data.read_data()
    # obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    # dataset = 'faa1'
    # task = 't4'
    # users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
    # greedy.set_data(users, all_attrs, priors, final)
    #
    # #Adaptive Epsilon-Greedy
    # epoch = 10
    # k = 3
    # epsilon = 0.05
    # l = 5
    # f = 1
    # f1_score = f1_before = f1_after = 0
    # for user in greedy.users:
    #     avrg_user = avg_userb = avg_usera = 0
    #     data = obj.read_cur_data(user, dataset)
    #     for experiment in range(epoch):
    #         accu_b, accu_a, accu = greedy.run_adaptive_epsilon_greedy(data, user, dataset, task, k, epsilon, l, f, False, True)
    #         avrg_user += accu
    #         avg_userb += accu_b
    #         avg_usera += accu_a
    #     f1_score += (avrg_user / epoch)
    #     f1_before += (avg_userb / epoch)
    #     f1_after += (avg_usera / epoch)
    # f1_score /= len(greedy.users)
    # f1_before /= len(greedy.users)
    # f1_after /= len(greedy.users)
    # print("l {} f {} F1 Score Adaptive Epsilon Greedy (no-state) ={} {} {}".format(l, f, f1_before, f1_after, f1_score))

    #Classical Epsilon-Greedy
    # epoch = 10
    # k = 3
    # epsilon = 0.05
    # states = [False]
    # for state in states:
    #     f1_score = 0
    #     f1_before = f1_after = 0
    #     for user in greedy.users:
    #         avrg_user = avg_userb = avg_usera = 0
    #         data = obj.read_cur_data(user, dataset)
    #         for experiment in range(epoch):
    #             accu_b, accu_a, accu = greedy.run_epsilon_greedy(data, user, dataset, task, k, epsilon, state, True)
    #             avrg_user += accu
    #             avg_userb += accu_b
    #             avg_usera += accu_a
    #         f1_score += (avrg_user / epoch)
    #         f1_before += (avg_userb / epoch)
    #         f1_after += (avg_usera / epoch)
    #     f1_score /= len(greedy.users)
    #     f1_before /= len(greedy.users)
    #     f1_after /= len(greedy.users)
    #     print("Epsilon {} Greedy: F1 Score ={} {} {}".format(epsilon, f1_before, f1_after, f1_score))
