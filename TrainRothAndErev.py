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

    def set_data(self, users, all_attrs, priors, final):
        self.users = users
        self.all_attrs = all_attrs
        self.priors = priors
        self.final = final

    #Roth and Erev algorithm without states
    def run_roth_and_erev(self, cur_data, user, dataset, cur_task, k, forgetting, cat = True):
        e = Evaluators.Evaluators()
        rae = modified_roth_and_erev(False)
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
        f1_score = 0
        no_of_intr = 0
        # count_attributes = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)

            # Getting the states
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

                #Payoff is calculated based on the number of *correct attributes* in the current interaction
                payoff = 0
                for attrs in picked_attr:
                    if attrs in self.final:
                        payoff += 1

                rae.update_qtable(user, ground, payoff, forgetting)
                # else:
                #     except_ground = self.priors.copy()
                #     for attr in ground:
                #         idx = except_ground.index(attr)
                #         del except_ground[idx]
                #     rae.update_qtable(user, except_ground, 10, 0.1)

                # print("interaction {}: ".format(no_of_intr))
                if no_of_intr >= 2:
                    # print(ground)
                    # print(picked_attr)
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
                    f1_score += get_f1
                    total += 1
                no_of_intr += 1

        f1_score = f1_score / total
        return f1_score

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
        f1_score = 0
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
    roth = TrainRothAndErev()
    roth.set_data(users, all_attrs, priors, final)

    f1_score = 0
    epoch = 20
    k = 3
    # forgetting = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # for ff in forgetting:
    ff  = 0.1
    f1_score = 0
    for user in roth.users:
        avrg_user = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu = roth.run_roth_and_erev_state(data, user, dataset, task, k, ff, True)
            # print("User: {} Precision@K {}".format(user, accu))
            avrg_user += accu
            # break
        f1_score += (avrg_user / epoch)
        # break
    f1_score /= len(roth.users)
    print("Forgetting: {} F1 Score (No State) = {}".format(ff, f1_score))

    f1_score = 0
    epoch = 20
    k = 3

    for user in roth.users:
        avrg_user = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu = roth.run_roth_and_erev_state(data, user, dataset, task, k,ff, True)
            # print("User: {} Precision@K {} ".format(user, accu))
            avrg_user += accu
            # break
        f1_score += (avrg_user / epoch)
        # break
    f1_score /= len(roth.users)
    print("F1 Score (With State) = {}".format(f1_score))