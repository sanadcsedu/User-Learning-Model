from collections import defaultdict
from modified_roth_and_erev import modified_roth_and_erev
import queue
import Categorizing
import Evaluators
import pdb
import read_data


class TrainRothAndErev:
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

    #Roth and Erev algorithm without states
    def run_roth_and_erev(self, cur_data, user, dataset, cur_task, k, forgetting, cat = True):
        e = Evaluators.Evaluators()
        rae = modified_roth_and_erev(False)

        rae.add_prior_strategies(user, self.all_attrs, 0)
        rae.add_prior_strategies(user, self.priors, 1)

        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        total = 0
        f1_score = []
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = rae.make_choice_nostate(user, k)
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)
                if cat:
                    ground = c.get_category(ground)

                if len(ground) == 0:
                    continue

                #reinforcing the learning model
                ##########################################################
                payoff = 0
                for attrs in ground:
                    if attrs in self.final:
                        payoff += 1
                rae.update_qtable(user, ground, payoff, forgetting)

                payoff = 0
                for attrs in picked_attr:
                    if attrs in ground:
                        payoff += 1
                rae.update_qtable(user, picked_attr, payoff, forgetting)
                ###########################################################

                if no_of_intr >= 2:
                    # print(ground)
                    # print(picked_attr)
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1

        return e.before_after(f1_score, total, self.threshold)

    # Running Roth and Erev algorithm with states
    def run_roth_and_erev_state(self, cur_data, user, dataset, cur_task, k, forgetting, cat = True):
        e = Evaluators.Evaluators()
        rae = modified_roth_and_erev(True)
        rae.add_prior_strategies(user, self.priors, 1)

        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        total = 0
        f1_score = []
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0
        count_attributes = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)

            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = rae.make_choice_state(user, list(prev_interactions.queue), k)
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

                rae.update_qtable(user, list(prev_interactions.queue), payoff, forgetting)

                if no_of_intr >= 2:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    # f1_score += get_f1
                    f1_score.append(get_f1)
                    total += 1
                no_of_intr += 1

        # Check the f1 accuracy before/after certain threshold
        threshold = int(total * self.threshold)
        f1_before = f1_after = 0
        cnt1 = cnt2 = 0
        for idx in range(len(f1_score)):
            if idx < threshold:
                f1_before += f1_score[idx]
                cnt1 += 1
            else:
                f1_after += f1_score[idx]
                cnt2 += 1
        f1_before /= threshold
        f1_after /= (len(f1_score) - threshold)
        # pdb.set_trace()
        return f1_before, f1_after
        # f1_score = f1_score / total
        # return f1_score

    def hyperparameter(self):
        obj = read_data.read_data()
        obj.create_connection(r"D:\Tableau Learning\Tableau.db")
        dataset = 'birdstrikes1'
        task = 't4'
        users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
        self.set_data(users, all_attrs, priors, final)

        epoch = 10
        k = 3
        forgetting = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        best_forget = _max = -1
        for ff in forgetting:
            f1_score = 0
            for user in self.users:
                avrg_user = 0
                data = obj.read_cur_data(user, dataset)
                for experiment in range(epoch):
                    accu_b, accu_a, accu = self.run_roth_and_erev(data, user, dataset, task, k, ff, True)
                    # print("User: {} Precision@K {}".format(user, accu))
                    avrg_user += accu
                f1_score += (avrg_user / epoch)
            f1_score /= len(self.users)
            if _max < f1_score:
                _max = f1_score
                best_forget = ff
        print("Roth and Erev F1 Score (No State) = {:.2f} -> Forgetting = {:.2f}".format(_max, best_forget))

    def f1_data(self, obj, dataset, task, epoch, k, ff):
        f1_score = f1_before = f1_after = 0
        for user in self.users:
            avrg_user = 0
            avg_userb = avg_usera = 0
            data = obj.read_cur_data(user, dataset)
            for experiment in range(epoch):
                accu_b, accu_a, accu = self.run_roth_and_erev(data, user, dataset, task, k, ff, True)
                avrg_user += accu
                avg_userb += accu_b
                avg_usera += accu_a
            f1_score += (avrg_user / epoch)
            f1_before += (avg_userb / epoch)
            f1_after += (avg_usera / epoch)
        f1_score /= len(self.users)
        f1_before /= len(self.users)
        f1_after /= len(self.users)
        print("Task {} f1_score, f1_before, f1_after = {:.2f} [{:.2f}, {:.2f}]".format(task, f1_score, f1_before, f1_after))

if __name__ == '__main__':
    # roth = TrainRothAndErev()
    # roth.hyperparameter()

    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = ['birdstrikes1', 'weather1', 'faa1']
    task = ['t2', 't3', 't4']

    epoch = 10
    k = 3
    ff = [0.25, 0.25, 0.25]
    print("***** F1-score Roth and Erev no-state *****")
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            roth = TrainRothAndErev()
            roth.set_data(users, all_attrs, priors, final)
            roth.f1_data(obj, d, task[idx], epoch, k, ff[idx])