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
        self.threshold = 0.4

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
        # Setting up action set for the Bush and Mosteller model
        action_set = []
        for cs in self.all_attrs:
            action_set.append("add+" + str(cs))
        for cs in self.all_attrs:
            action_set.append("drop+" + str(cs))
        action_set.append("reset")
        action_set.append("unchanged")
        rbm.add_prior_strategies(action_set)

        # Assigning some probabilities based on prior
        prior_set = []
        for cs in self.priors:
            prior_set.append("add+" + str(cs))
        rbm.add_prior_strategies(prior_set)

        total = 0
        f1_score = []
        no_of_intr = 0
        prev_attrs = []
        prev_states = []

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:

                # if state == prev_states:
                #     continue
                # else:
                #     prev_states = state

                if len(states) >= 1:
                    picked_action = rbm.make_choice_nostate(k)

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

                    # reinforcing the learning model
                    ##########################################################
                    # if picked_action[0] in cur_action:
                    #     # rbm.update(user, cur_action, 1)
                    #     if action == "add":
                    #         # rbm.update(user, cur_action, -1)
                    #         cur_action = []
                    #         for indx in range(len(new_attrs)):
                    #             cur_action.append("drop+" + new_attrs[indx])
                    #         rbm.update(user, cur_action, 1)
                    #     elif action == "drop":
                    #         # rbm.update(user, cur_action, -1)
                    #     else:
                    #         rbm.update(user, cur_action, 1)
                    #
                    # else:
                        # rbm.update(user, cur_action, 1)
                    if action == "add":
                        # rbm.update(user, cur_action, -1)
                        cur_action = []
                        for indx in range(len(new_attrs)):
                            cur_action.append("drop+" + new_attrs[indx])
                        rbm.update(user, cur_action, 1)
                    # elif action == "drop":
                        # rbm.update(user, cur_action, -1)
                    else:
                        rbm.update(user, cur_action, 1)
                        # rbm.update(user, picked_action, -1)
                    ###########################################################

                    if no_of_intr >= 2:
                        # print("interaction: {}".format(cur_attrs))
                        # print("Picked: {}".format(picked_action))
                        # print("Cur Action: {}".format(cur_action))
                        ############################
                        # flag = True
                        # for a in cur_action:
                        #     if a not in picked_action:
                        #         flag = False
                        #         break
                        # if flag:
                        #     get_f1 = 1
                        # else:
                        #     get_f1 = 0
                        ############################
                        # _, _, get_f1 = e.f1_score(cur_action, picked_action)
                        ############################
                        #calculating Recall (contains partial credits)
                        get_f1 = 0
                        for a in cur_action:
                            if a in picked_action:
                                get_f1 += 1
                        get_f1 /= len(cur_action)
                        # print(get_f1)
                        ############################
                        f1_score.append(get_f1)
                        total += 1
                    no_of_intr += 1

        if total <= 5:
            return -1, -1, -1
        else:
            return e.before_after(f1_score, total, self.threshold)

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
        f1_score = []
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
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        best_alpha = best_beta = _max = -1
        for alpha in alphas:
            for beta in betas:
                f1_score = 0
                for user in self.users:
                    avrg_user = 0
                    data = obj.read_cur_data(user, dataset)
                    for experiment in range(epoch):
                        accu_b, accu_a, accu = self.run_bush_mosteller_nostate(user, data, dataset, task, k, alpha, beta, True)
                        avrg_user += accu
                        # print("User {} accu {}".format(user, accu))
                    f1_score += (avrg_user / epoch)
                f1_score /= len(self.users)
                if _max < f1_score:
                    _max = f1_score
                    best_alpha = alpha
                    best_beta = beta
        print("Bush and Mosteller f1-score (no-state) {} -> Alpha {} Beta {}".format(_max, best_alpha, best_beta))

    def f1_data(self, obj, dataset, task, epoch, k, alpha, beta):
        f1_score = f1_before = f1_after = 0
        num_users = bush.users
        minus = 0
        for user in num_users:
            avrg_user = 0
            avg_userb = avg_usera = 0
            data = obj.read_cur_data(user, dataset)
            flag = True
            for experiment in range(epoch):
                accu_b, accu_a, accu = self.run_bush_mosteller_nostate(user, data, dataset, task, k, alpha, beta, True)
                if accu == -1:
                    flag = False
                    break
                avrg_user += accu
                avg_userb += accu_b
                avg_usera += accu_a
            if flag is False:
                minus += 1
                continue
            f1_score += (avrg_user / epoch)
            f1_before += (avg_userb / epoch)
            f1_after += (avg_usera / epoch)
        f1_score /= len(num_users)
        f1_before /= len(num_users)
        f1_after /= len(num_users)
        print("Task {} f1_score, f1_before, f1_after = {:.2f} [{:.2f}, {:.2f}]".format(task, f1_score, f1_before, f1_after))

if __name__ == '__main__':
    # bush = TrainingBushMosteller()
    # bush.hyperparameter()

    obj = read_data.read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    dataset = ['birdstrikes1', 'weather1', 'faa1']
    task = ['t2', 't3', 't4']

    epoch = 10
    k = 3
    #Hyper-parameter values for task t2, t3 and t4
    alpha = [0.5, 0.5, 0.5]
    beta = [0.1, 0.15, 0.05]
    print("***** F1-score Bush and Mosteller no-state *****")
    for d in dataset:
        print("Dataset: {}".format(d))
        print("###########################")
        for idx in range(len(task)):
            users, all_attrs, priors, final = obj.TableauDataset(d, task[idx])
            bush = TrainingBushMosteller()
            bush.set_data(users, all_attrs, priors, final)
            bush.f1_data(obj, d, task[idx], epoch, k, alpha[idx], beta[idx])
