import numpy as np
from collections import defaultdict
from no_exploration import no_exploration
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
        self.threshold = 0.5

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
        epg = no_exploration(action_set, epsilon, 0, 0, state)

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

        update_threshold = 0
        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            if task == cur_task:
                update_threshold += 1
        update_threshold = int(update_threshold * 0.5)
        # print(update_threshold)

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
                if no_of_intr < update_threshold:
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
                    _, _, get_f1 = e.f1_score(cur_action, picked_action)
                    # print("Ground: {}".format(cur_action))
                    # print("Picked: {}".format(picked_action))

                    # calculating Recall (contains partial credits)
                    # get_f1 = 0
                    # for a in cur_action:
                    #     if a in picked_action:
                    #         get_f1 += 1
                    # get_f1 /= len(cur_action)

                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1
        # x, y, z = e.before_after(f1_score, total, self.threshold)
        # print("{:.2f} {:.2f} {:.2f}".format(x, y, z))
        return e.before_after(f1_score, total, self.threshold)

    def f1_data(self, obj, dataset, task, epoch, k, epsilon, l, f, state = False, cat = True):
        f1_score = f1_before = f1_after = 0
        for user in self.users:
            avrg_user = avg_userb = avg_usera = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                accu_b, accu_a, accu = self.run_epsilon_greedy(data, user, dataset, task, k, epsilon, state, cat)
                # print("-> {:.2f} {:.2f} {:.2f}".format(accu_b, accu_a, accu))
                avrg_user += accu
                avg_userb += accu_b
                avg_usera += accu_a
            f1_score += (avrg_user / epoch)
            f1_before += (avg_userb / epoch)
            f1_after += (avg_usera / epoch)
        # print("-> {:.2f} {:.2f} {:.2f}".format(f1_before, f1_after, f1_score))
        f1_score /= len(self.users)
        f1_before /= len(self.users)
        f1_after /= len(self.users)
        print("Task {} f1_score, f1_before, f1_after = {:.2f} [{:.2f}, {:.2f}]".format(task, f1_score, f1_before,
                                                                                       f1_after))

if __name__ == '__main__':

    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = ['birdstrikes1', 'weather1', 'faa1']
    task = ['t2', 't3', 't4']

    epoch = 20
    k = 3
    epsilon = [0.15, 0.05, 0.05]

    # Classical Epsilon-Greedy
    print("***** No-Exploration *****")
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            greedy = TrainGreedy()
            greedy.set_data(users, all_attrs, priors, final)
            greedy.f1_data(obj, d, task[idx], epoch, k, epsilon[idx], None, None)