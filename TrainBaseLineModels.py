import Categorizing
import queue
import Evaluators
import pdb
import numpy as np
import read_data


class BaseLineModels:

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

    def run_fixed_strategy(self, user, cur_data, dataset, cur_task, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            elif dataset == 'weather1':
                c.weather1()
            else:
                c.faa1()

        #Choosing actions based on prior
        #In our case the attributes are picked based on the attirbutes listed in the given questions
        #For task 4 (open-ended), the prior attributes are collected from T2 and T3.
        fixed_attrs = []
        if dataset == 'birdstrikes1':
            if cur_task == 't2':
                fixed_attrs = ['"ac_class"', '"damage"', '"number of records"']
            elif cur_task == 't3':
                fixed_attrs = ['"precip"', '"sky"', '"incident_date"', '"number of records"']
            elif cur_task == 't4':
                fixed_attrs = ['"ac_class"', '"damage"', '"precip"', '"sky"', '"incident_date"', '"number of records"']
        elif dataset == 'faa1':
            if cur_task == 't2':
                fixed_attrs = ['"flightdate"', '"uniquecarrier"', '"number of records"']
            elif cur_task == 't3':
                fixed_attrs = ['"arrdelay"', '"distance"', '"number of records"']
            elif cur_task == 't4':
                fixed_attrs = ['"flightdate"', '"uniquecarrier"', '"arrdelay"', '"distance"', '"number of records"']
        else:
            if cur_task == 't2':
                fixed_attrs = ['"tmax"', '"tmin"', '"date"', '"number of records"']
            elif cur_task == 't3':
                fixed_attrs = ['"highwinds"', '"state"', '"number of records"']
            elif cur_task == 't4':
                fixed_attrs = ['"tmax"', '"tmin"', '"date"', '"highwinds"', '"state"', '"number of records"']

        total = 0
        f1_score = []
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                if len(states) >= 1:
                    picked_attr = c.get_category(fixed_attrs)
                    ground = []
                    for s in states:
                        if len(s) >= 1:
                            ground.append(s)

                    if cat:
                        ground = c.get_category(ground)

                    if len(ground) == 0:
                        continue

                    if no_of_intr >= 2:
                        # print("Ground {}".format(ground))
                        # print("Picked {}".format(picked_attr))
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        # f1_score += get_f1
                        f1_score.append(get_f1)
                        total += 1
                    no_of_intr += 1

        return e.before_after(f1_score, total, self.threshold)

    def randomized(self, choices, k):
        ret = []
        while k > 0:
            pick = np.random.randint(0, len(choices))
            ret.append(choices[pick])
            del choices[pick]
            k -= 1
        return ret

    def run_randomized(self, user, cur_data, dataset, cur_task, k, cat = False):
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
        f1_score = []
        no_of_intr = 0

        for row in cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            if task == cur_task:
                picked_attr = self.randomized(self.priors.copy(), k)
                ground = []
                for s in states:
                    if len(s) >= 1:
                        ground.append(s)

                if cat:
                    ground = c.get_category(ground)
                if len(ground) == 0:
                    continue

                if no_of_intr >= 2:
                    _, _, get_f1 = e.f1_score(ground, picked_attr)
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
    baseline = BaseLineModels()
    baseline.set_data(users, all_attrs, priors, final)

    f1_score = f1_before = f1_after = 0
    epoch = 10
    for user in baseline.users:
        avrg_user = 0
        avg_userb = avg_usera = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu_b, accu_a, accu = baseline.run_fixed_strategy(user, data, dataset, task, True)
            avrg_user += accu
            avg_userb += accu_b
            avg_usera += accu_a
            # print("User {} accu {}".format(user, accu))
        f1_score += (avrg_user / epoch)
        f1_before += (avg_userb / epoch)
        f1_after += (avg_usera / epoch)
    f1_score /= len(baseline.users)
    f1_before /= len(baseline.users)
    f1_after /= len(baseline.users)
    print("f1 score Fixed-strategy (nostate) = {:.2f} [{:.2f}, {:.2f}]".format(f1_score, f1_before, f1_after))

    k = 3
    f1_score = f1_before = f1_after = 0
    for user in baseline.users:
        avrg_user = 0
        avg_userb = avg_usera = 0
        data = obj.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu_b, accu_a, accu = baseline.run_randomized(user, data, dataset, task, k, True)
            # print("User {} accu {}".format(user, accu))
            avrg_user += accu
            avg_userb += accu_b
            avg_usera += accu_a
        f1_score += (avrg_user / epoch)
        f1_before += (avg_userb / epoch)
        f1_after += (avg_usera / epoch)
    f1_score /= len(baseline.users)
    f1_before /= len(baseline.users)
    f1_after /= len(baseline.users)
    print("f1 score Randomized (nostate) = {:.2f} [{:.2f}, {:.2f}]".format(f1_score, f1_before, f1_after))