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
        # epg = EpsilonGreedy(self.all_attrs, epsilon, 0, 0, state)
        # epg.update([self.priors], 1)

        # Setting up action set for the Epsilon-Greedy Algorithm
        action_set = []
        for cs in self.all_attrs:
            action_set.append("add+" + str(cs))
        for cs in self.all_attrs:
            action_set.append("drop+" + str(cs))
        action_set.append("reset")
        action_set.append("unchanged")
        epg = EpsilonGreedy(action_set, epsilon, 0, 0, state)

        # Assigning some probabilities based on prior
        # prior_set = []
        # for cs in self.priors:
        #     prior_set.append("add+" + str(cs))
        # # pdb.set_trace()
        # epg.update([prior_set], 1)

        f1_score = []
        prev_interactions = None # Because no-state
        no_of_intr = 0
        prev_attrs = []

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_action = epg.make_choice_classic(k, prev_interactions)

                # Figuring out which action has been performed
                cur_attrs = []
                for s in states:
                    if len(s) >= 1:
                        cur_attrs.append(s)
                if cat:
                    cur_attrs = c.get_category(cur_attrs)
                action = None
                if len(cur_attrs) == 0:
                    action = "reset"
                elif cur_attrs == prev_attrs:
                    action = "unchanged"
                elif len(prev_attrs) == len(cur_attrs):
                    # Check if the interaction is equal
                    action = "unchanged"
                    for attrs in prev_attrs:
                        if attrs not in cur_attrs:
                            action = "replace"
                            break
                    if action == "unchanged":
                        continue

                    # pdb.set_trace()
                    # replace can be considered as a combo of existing actions. 1. Drop attributes 2. Add attirbutes
                    dropped = []
                    for attrs in prev_attrs:
                        if attrs not in cur_attrs:
                            dropped.append(attrs)
                    new_attrs = []
                    for attrs in cur_attrs:
                        if attrs not in prev_attrs:
                            new_attrs.append(attrs)
                else:
                    if len(prev_attrs) < len(cur_attrs):  # new attribute has been added
                        # Find out the new attribute that has been added
                        new_attrs = []
                        for attrs in cur_attrs:
                            if attrs not in prev_attrs:
                                new_attrs.append(attrs)
                        action = "add"
                    elif len(prev_attrs) > len(cur_attrs):  # attribute has been deleted
                        # Find out which attributes has been deleted
                        dropped = []
                        for attrs in prev_attrs:
                            if attrs not in cur_attrs:
                                dropped.append(attrs)
                        action = "drop"

                prev_attrs = cur_attrs
                cur_action = []
                if action == "add":
                    for indx in range(len(new_attrs)):
                        cur_action.append("add+" + new_attrs[indx])
                elif action == "drop":
                    for indx in range(len(dropped)):
                        cur_action.append("drop+" + dropped[indx])
                elif action == "replace":
                    for indx in range(len(new_attrs)):
                        cur_action.append("add+" + new_attrs[indx])
                else:
                    cur_action.append(action)

                # epg.update([cur_action], 1)
                if action == "add":
                    # rae.update_qtable(user, cur_action, 1, forgetting)
                    cur_action = []
                    for indx in range(len(new_attrs)):
                        cur_action.append("drop+" + new_attrs[indx])
                    epg.update([cur_action], 1)
                elif action == "drop":
                    epg.update([cur_action], 1)
                else:
                    epg.update([cur_action], 1)

                if no_of_intr >= after:
                    # _, _, get_f1 = e.f1_score(cur_action, picked_action)

                    # calculating Recall (contains partial credits)
                    get_f1 = 0
                    for a in cur_action:
                        if a in picked_action:
                            get_f1 += 1
                    get_f1 /= len(cur_action)

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
        # aepg = EpsilonGreedy(self.all_attrs, epsilon, l, f, state)
        # aepg.update([self.priors], 1)
        # Setting up action set for the Epsilon-Greedy Algorithm
        action_set = []
        for cs in self.all_attrs:
            action_set.append("add+" + str(cs))
        for cs in self.all_attrs:
            action_set.append("drop+" + str(cs))
        action_set.append("reset")
        action_set.append("unchanged")
        aepg = EpsilonGreedy(action_set, epsilon, l, f, state)

        # Assigning some probabilities based on prior
        prior_set = []
        for cs in self.priors:
            prior_set.append("add+" + str(cs))
        aepg.update([prior_set], 1)

        f1_score = []
        prev_interactions = None #For now no-state
        no_of_intr = 0
        prev_attrs = []

        for row in cur_data:
            userid, task, seqid, state = tuple(row)

            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_action = aepg.make_choice_adaptive(k, None)

                # Figuring out which action has been performed
                cur_attrs = []
                for s in states:
                    if len(s) >= 1:
                        cur_attrs.append(s)
                if cat:
                    cur_attrs = c.get_category(cur_attrs)
                action = None
                if len(cur_attrs) == 0:
                    action = "reset"
                elif cur_attrs == prev_attrs:
                    action = "unchanged"
                elif len(prev_attrs) == len(cur_attrs):
                    # Check if the interaction is equal
                    action = "unchanged"
                    for attrs in prev_attrs:
                        if attrs not in cur_attrs:
                            action = "replace"
                            break
                    if action == "unchanged":
                        continue

                    # pdb.set_trace()
                    # replace can be considered as a combo of existing actions. 1. Drop attributes 2. Add attirbutes
                    dropped = []
                    for attrs in prev_attrs:
                        if attrs not in cur_attrs:
                            dropped.append(attrs)
                    new_attrs = []
                    for attrs in cur_attrs:
                        if attrs not in prev_attrs:
                            new_attrs.append(attrs)
                else:
                    if len(prev_attrs) < len(cur_attrs):  # new attribute has been added
                        # Find out the new attribute that has been added
                        new_attrs = []
                        for attrs in cur_attrs:
                            if attrs not in prev_attrs:
                                new_attrs.append(attrs)
                        action = "add"
                    elif len(prev_attrs) > len(cur_attrs):  # attribute has been deleted
                        # Find out which attributes has been deleted
                        dropped = []
                        for attrs in prev_attrs:
                            if attrs not in cur_attrs:
                                dropped.append(attrs)
                        action = "drop"

                prev_attrs = cur_attrs
                cur_action = []
                if action == "add":
                    for indx in range(len(new_attrs)):
                        cur_action.append("add+" + new_attrs[indx])
                elif action == "drop":
                    for indx in range(len(dropped)):
                        cur_action.append("drop+" + dropped[indx])
                elif action == "replace":
                    for indx in range(len(new_attrs)):
                        cur_action.append("add+" + new_attrs[indx])
                else:
                    cur_action.append(action)

                # aepg.update([cur_action], 1)
                #####
                if action == "add":
                    # rae.update_qtable(user, cur_action, 1, forgetting)
                    cur_action = []
                    for indx in range(len(new_attrs)):
                        cur_action.append("drop+" + new_attrs[indx])
                    aepg.update([cur_action], 1)
                elif action == "drop":
                    aepg.update([cur_action], 1)
                else:
                    aepg.update([cur_action], 1)
                #####
                if no_of_intr >= after:
                    ########################
                    # calculating Recall (contains partial credits)
                    get_f1 = 0
                    for a in cur_action:
                        if a in picked_action:
                            get_f1 += 1
                    get_f1 /= len(cur_action)
                    ########################
                    # _, _, get_f1 = e.f1_score(cur_action, picked_action)
                    ########################
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1
        # pdb.set_trace()
        return e.before_after(f1_score, total, self.threshold)

    def hyperparameter_classic(self):
        obj = read_data.read_data()
        obj.create_connection(r"D:\Tableau Learning\Tableau.db")
        dataset = 'birdstrikes1'
        task = 't2'
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

        print(" Epsilon {} Greedy (no state) -> F1 Score= {:.2f}".format(best_epsilon, _max))

    def hyperparameter_adaptive(self):
        obj = read_data.read_data()
        obj.create_connection(r"D:\Tableau Learning\Tableau.db")
        dataset = 'birdstrikes1'
        task = 't2'
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
        print("l {} f {} F1 Score Adaptive Epsilon Greedy (no-state) = {:.2f}".format(best_l, best_f, _max))

    def f1_data(self, adaptive, obj, dataset, task, epoch, k, epsilon, l, f, state = False, cat = True):
        f1_score = f1_before = f1_after = 0
        for user in self.users:
            avrg_user = avg_userb = avg_usera = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                if adaptive:
                    accu_b, accu_a, accu = self.run_adaptive_epsilon_greedy(data, user, dataset, task, k, epsilon, l, f, state, cat)
                else:
                    accu_b, accu_a, accu = self.run_epsilon_greedy(data, user, dataset, task, k, epsilon, state, cat)
                avrg_user += accu
                avg_userb += accu_b
                avg_usera += accu_a
            f1_score += (avrg_user / epoch)
            f1_before += (avg_userb / epoch)
            f1_after += (avg_usera / epoch)
        f1_score /= len(self.users)
        f1_before /= len(self.users)
        f1_after /= len(self.users)
        print("Task {} f1_score, f1_before, f1_after = {:.2f} [{:.2f}, {:.2f}]".format(task, f1_score, f1_before,
                                                                                       f1_after))


if __name__ == '__main__':
    # greedy = TrainGreedy()
    # greedy.hyperparameter_classic()
    # greedy.hyperparameter_adaptive()

    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = ['birdstrikes1', 'weather1', 'faa1']
    task = ['t2', 't3', 't4']

    #Adaptive Epsilon-Greedy
    print("***** F1-score Adaptive Epsilon-Greedy no-state *****")
    epoch = 10
    k = 3
    epsilon = [0.15, 0.05, 0.05]
    l = [8, 5, 6]
    f = [4, 2, 2]
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            greedy = TrainGreedy()
            greedy.set_data(users, all_attrs, priors, final)
            greedy.f1_data(True, obj, d, task[idx], epoch, k, epsilon[idx], l[idx], f[idx])

    # Classical Epsilon-Greedy
    print("***** F1-score Classical Epsilon-Greedy no-state *****")
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            greedy = TrainGreedy()
            greedy.set_data(users, all_attrs, priors, final)
            greedy.f1_data(False, obj, d, task[idx], epoch, k, epsilon[idx], None, None)