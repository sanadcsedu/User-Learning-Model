from collections import defaultdict
from LatestReward import LatestReward
from WinKeepLoseRandomize import WinKeepLoseRandomize
import Categorizing
import Evaluators
import read_data
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
        sim = WinKeepLoseRandomize(self.all_attrs, self.final)
        after = 2
        f1_score = []
        no_of_intr = 0
        for row in cur_data:
            userid, ttask, seqid, state = tuple(row)
            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')

            if ttask == task:
                picked_attr = sim.make_choice(k)
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)
                if cat:
                    ground = c.get_category(ground)
                if len(ground) == 0:
                    continue

                sim.assign_reward(ground)

                if no_of_intr >= after:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    # print("Ground: {}".format(ground))
                    # print("Picked: {}".format(picked_attr))
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1
        return e.before_after(f1_score, total, self.threshold)

if __name__ == '__main__':
    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = 'faa1'
    task = 't4'
    users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
    heu = TrainHeuristics()
    heu.set_data(users, all_attrs, priors, final)

    f1_score = f1_before = f1_after = 0
    epoch = 10
    k = 3
    for user in heu.users:
        avrg_user = avg_userb = avg_usera = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu_b, accu_a, accu = heu.WinKeepLoseRandomize(data, dataset, task, k, True)
            avrg_user += accu
            avg_userb += accu_b
            avg_usera += accu_a
        f1_score += (avrg_user / epoch)
        f1_before += (avg_userb / epoch)
        f1_after += (avg_usera / epoch)
    f1_score /= len(heu.users)
    f1_before /= len(heu.users)
    f1_after /= len(heu.users)
    print("F1 Score Win-Keep Lose-Randomize (nostate) = {:.2f} [{:.2f}, {:.2f}]".format(f1_score, f1_before, f1_after))

    # f1_before = f1_after = 0
    # epoch = 10
    # k = 3
    # for user in heu.users:
    #     avg_userb = avg_usera = 0
    #     data = obj.read_cur_data(user, dataset)
    #     for experiment in range(epoch):
    #         accu_b, accu_a = heu.LatestReward(data, dataset, task, k, True)
    #         avg_userb += accu_b
    #         avg_usera += accu_a
    #     f1_before += (avg_userb / epoch)
    #     f1_after += (avg_usera / epoch)
    # f1_before /= len(heu.users)
    # f1_after /= len(heu.users)
    # print("F1 Score Latest-Reward (nostate) = {} {}".format(f1_before, f1_after))
