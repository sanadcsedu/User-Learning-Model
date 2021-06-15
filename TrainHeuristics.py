from collections import defaultdict
from LatestReward import LatestReward
from WinKeepLoseRandomize import WinKeepLoseRandomize
import Categorizing
import Evaluators
import read_data
import numpy as np
import queue
import pdb

#Main goal of this class is to find out which model replicates User Learning perfectly
#It tests heuristic models such as "Latest-Reward" and "Win-Keep Lose-Randomize"
#These models are relatively simple and popular in Game Theory and Behavioral Economics community


class TrainHeuristics:

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

    #Running Latest-Reward
    def LatestReward(self, cur_data, dataset, task, k, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        base = LatestReward(all_attrs, final, 1)
        f1_score = []
        no_of_intr = 0
        total = 0
        after = 2
        for row in cur_data:
            userid, ttask, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if ttask == task:
                picked_attr = base.make_choice(k)
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)

                if cat:
                    ground = c.get_category(ground)
                if len(ground) == 0:
                    continue

                base.assign_reward()
                if no_of_intr >= after:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1

        return e.before_after(f1_score, total, self.threshold)


    #Running Win-Keep Lose-Randomize
    def WinKeepLoseRandomize(self, cur_data, dataset, task, k, cat = False):
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
        # sim = WinKeepLoseRandomize(self.all_attrs, self.final)
        action_set = []
        for cs in self.all_attrs:
            action_set.append("add+"+ str(cs))
        for cs in self.all_attrs:
            action_set.append("drop+" + str(cs))
        action_set.append("reset")
        action_set.append("unchanged")

        # pdb.set_trace()

        after = 2
        f1_score = []
        no_of_intr = 0
        prev_action = ["reset"]
        prev_attrs = []
        unchgcnt = 0

        for row in cur_data:
            userid, ttask, seqid, state = tuple(row)
            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')

            if ttask == task:
                picked_action = prev_action

                #Figuring out which action has been performed
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
                    unchgcnt += 1
                elif len(prev_attrs) == len(cur_attrs):
                    #Check if the interaction is equal
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
                    if len(prev_attrs) < len(cur_attrs): #new attribute has been added
                        #Find out the new attribute that has been added
                        new_attrs = []
                        for attrs in cur_attrs:
                            if attrs not in prev_attrs:
                                new_attrs.append(attrs)
                        action = "add"
                    elif len(prev_attrs) > len(cur_attrs): #attribute has been deleted
                        #Find out which attributes has been deleted
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

                if no_of_intr >= after:
                    # _, _, get_f1 = e.f1_score(ground, picked_attr)
                    # print("interaction: {}".format(cur_attrs))
                    # print("Ground: {}".format(cur_action))
                    # print("Picked: {}".format(picked_action))

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

                # temp = False
                # for a in cur_action:
                #     if a in picked_action:
                #         temp = True
                if cur_action == picked_action:
                    prev_action = cur_action
                # if temp:
                #     prev_action = cur_action
                else:
                    ret = []
                    choices = action_set.copy()
                    kk = k
                    while kk > 0:
                        pick = np.random.randint(0, len(choices))
                        ret.append(choices[pick])
                        del choices[pick]
                        kk -= 1
                    # pick = np.random.randint(len(action_set))
                    prev_action = ret
                    # prev_action = cur_action
                # pdb.set_trace()
        # print(f1_score)
        # print(total)
        return e.before_after(f1_score, total, self.threshold)

    def f1_data(self, obj, dataset, task, epoch, k):
        f1_score = f1_before = f1_after = 0
        num_users = self.users
        for user in num_users:
            avrg_user = 0
            avg_userb = avg_usera = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                accu_b, accu_a, accu = self.WinKeepLoseRandomize(data, dataset, task, k, True)
                avrg_user += accu
                avg_userb += accu_b
                avg_usera += accu_a
            f1_score += (avrg_user / epoch)
            f1_before += (avg_userb / epoch)
            f1_after += (avg_usera / epoch)
        f1_score /= len(num_users)
        f1_before /= len(num_users)
        f1_after /= len(num_users)
        print("Task {} f1_score, f1_before, f1_after = {:.2f} [{:.2f}, {:.2f}]".format(task, f1_score, f1_before, f1_after))

if __name__ == '__main__':
    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")

    dataset = ['birdstrikes1', 'weather1', 'faa1']
    task = ['t2', 't3', 't4']

    epoch = 10
    k = 3

    print("***** F1-score Win-Keep Lose-Randomize no-state *****")
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            heu = TrainHeuristics()
            heu.set_data(users, all_attrs, priors, final)
            heu.f1_data(obj, d, task[idx], epoch, k)
