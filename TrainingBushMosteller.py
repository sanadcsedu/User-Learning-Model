from BushMosteller import BushMosteller
import Categorizing
import queue
import Evaluators
import pdb
import read_data


class TrainingBushMosteller:

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

    #Bush and Mosteller algorithm without states
    def run_bush_mosteller_nostate(self, user, cur_data, dataset, cur_task, k, alpha, beta, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        rbm = BushMosteller(alpha, beta, False)
        rbm.add_prior_strategies(self.priors)

        total = 0
        f1_score = 0
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                if len(states) >= 1:
                    picked_attr = rbm.make_choice_nostate(k)
                    ground = []
                    for s in states:
                        if len(s) >= 1:
                            ground.append(s)

                    if cat:
                        ground = c.get_category(ground)

                    if len(ground) == 0:
                        continue

                    # Payoff is calculated based on the number of *correct attributes* in the current interaction
                    cnt = 0
                    for attrs in picked_attr:
                        if attrs in self.final:
                            cnt += 1

                    if cnt > 0: #Positive reward if we find the interaction useful
                        rbm.update(user, ground, 1)
                    # else: #Negative reward if we find the interaction not useful
                    #     rbm.update(user, ground, -1)

                    if no_of_intr >= 2:
                        # print("Ground {}".format(ground))
                        # print("Picked {}".format(picked_attr))
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        f1_score += get_f1
                        total += 1
                    no_of_intr += 1

        f1_score = f1_score / total
        return f1_score

    #Bush and Mosteller algorithm with states
    def run_bush_mosteller_state(self,user, cur_data, dataset, cur_task, k, alpha, beta, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        rbm = BushMosteller(alpha, beta, True)
        rbm.add_prior_strategies(self.priors)

        total = 0
        f1_score = 0
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = rbm.make_choice_state(user, list(prev_interactions.queue), k)
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

                # Payoff is calculated based on the number of *correct attributes* in the current interaction
                cnt = 0
                for attrs in picked_attr:
                    if attrs in self.final:
                        cnt += 1

                if cnt > 0: #Positive reward if we find the interaction useful
                    rbm.update(user, list(prev_interactions.queue), 1)
                # else: #Negative reward if we find the interaction not useful
                #     rbm.update(user, list(prev_interactions.queue), -1)

                if no_of_intr >= 2:
                    # print("Ground {}".format(ground))
                    # print("Picked {}".format(picked_attr))
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
    task = 't4'
    users, all_attrs, priors, final = obj.TableauDataset(dataset, task)
    bush = TrainingBushMosteller()
    bush.set_data(users, all_attrs, priors, final)

    f1_score = 0
    epoch = 20
    k = 3
    # alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    alpha = 0.25
    beta = 0.2
    _max = -1
    # for alpha in alphas:
    #     for beta in betas:
    f1_score = 0
    for user in bush.users:
        avrg_user = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu = bush.run_bush_mosteller_nostate(user, data, dataset, task, k, alpha, beta, True)
            avrg_user += accu
            # print("User {} accu {}".format(user, accu))
        f1_score += (avrg_user / epoch)
    f1_score /= len(bush.users)
    # _max = max(_max, f1_score)
    print("A = {} B = {} f1 score Bush and Mosteller without state = {}".format(alpha, beta, f1_score))
    # print(_max)
    f1_score = 0
    for user in bush.users:
        avrg_user = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu = bush.run_bush_mosteller_state(user, data, dataset, task, k, alpha, beta, True)
            # print("User {} accu {}".format(user, accu))
            avrg_user += accu
        f1_score += (avrg_user / epoch)
    f1_score /= len(bush.users)
    print("f1 score Bush and Mosteller state-based = {}".format(f1_score))