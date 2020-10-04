import sqlite3
import re
import numpy as np
from collections import defaultdict
from BushMosteller import BushMosteller
import queue


class TrainingBushMosteller:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        # self.users = [1,5,9,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        self.users = [1,5,109,13,25,29,33,53,57,61,73,97]
        self.attr_dict = dict()
        self.attributes = defaultdict()

        #list of T4 attributes
        self.t4attributes = defaultdict()

    #Creating a connection with the database using sqlite3
    def create_connection(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file)
            # print(sqlite3.version)
        except sqlite3.Error as e:
            print(e)

    #Read all the information form the database and store important ones inside the class \ current user
    def read_data(self, userid):
        c = self.conn.cursor()
        # print(self.users)
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid <> ' + str(userid)
        c.execute(query)
        # c.execute('select distinct userid from master_file where dataset = \'birdstrikes1\'')
        self.data = c.fetchall()
        # print(self.data)
        # for row in self.data:
        #     print(row[0], end=",")

    # Read all the information form the database and store important ones inside the class for current user
    def read_cur_data(self, userid):
        c = self.conn.cursor()
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid = ' + str(userid)
        c.execute(query)
        self.cur_data = c.fetchall()

    #For peeking into the saved data (esp. the attributes) as tuples
    def read_states(self, user):
        # print("########## " + str(user) + " #############")
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            # state = state.replace('"', '')
            states = state.split(', ')
            # print(task, end=" ")
            for s in states:
                # print(s, end= " ")
                if s in self.attr_dict:
                    self.attr_dict[s] += 1
                else:
                    self.attr_dict[s] = 1

    # removes attributes with low appearances (<4) and creates a new attribute list
    def remove_attr(self):
        # print(len(self.attr_dict))
        for attr in self.attr_dict.copy():
            if self.attr_dict[attr] < 10 or len(attr) < 1:
                # print(attr)
                del self.attr_dict[attr]
            else:
                # print(attr, end = ", ")
                # print(self.attr_dict[attr])
                self.attributes[attr] = 1

    # Calculates Precision@k of predicted attributes against ground value
    def find_Precision_at_k(self, ground, test, k):
        found = 0
        for attr in ground:
            flag = 0
            for attr_test in test:
                if attr == attr_test:
                    flag = 1
            if flag == 1:
                found += 1
        precision_at_k = found / k
        return precision_at_k

    #Bayesian Learning for Individual User with Uniform Prior (Task 4)
    def run_bush_mosteller_v2(self, k, alpha = 0.2):

        rbm = BushMosteller(alpha)
        # Setting up some prior strategies based on Task 2 and Task 3
        priors = ['"incident_date"', '"precip"', '"sky"', '"birds_struck"', '"ac_class"']
        rbm.add_prior_strategies(priors)

        # Final query holds the attributes used in final query. So only this particular set of
        # of attributes will be rewarded
        final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
                 '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
        final_query = defaultdict()
        for every in final:
            final_query[every] = 1

        total = 0
        precision_at_k = 0
        no_of_intr = 0
        prev_interactions = queue.Queue(maxsize=2)
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            interactions = []
            # Getting the attributes used for Testing on T4
            if task == 't4':
                if len(states) >= 1:
                    picked_attr = rbm.make_choice(k)
                    test = []
                    for s in states:
                        if len(s) >= 1:
                            interactions.append(s)
                            test.append(s)

                    if len(interactions) < 1:
                        continue

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    # Payoff is calculated based on the number of *correct attributes* in the current
                    # interaction
                    cnt = 0
                    for attrs in picked_attr:
                        if attrs in final_query:
                            cnt += 1

                    if cnt > 0:
                        rbm.update(list(prev_interactions.queue))

                    #Calculate Precision after 1st 5 interactions
                    if no_of_intr >= 5:
                        # print("TEST")
                        # print(test)
                        # print(picked_attr)
                        precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                        total += 1
                    no_of_intr += 1

        # rae.tester(user)
        precision_at_k = precision_at_k / total
        return precision_at_k


    #Bayesian Learning for Individual User with Uniform Prior (Task 3)
    def run_bush_mosteller_v1(self, k, alpha = 0.2):

        rbm = BushMosteller(alpha)
        # Setting up some prior strategies based on Task 2 and Task 3
        priors = ['"incident_date"', '"precip"', '"sky"']
        rbm.add_prior_strategies(priors)

        # Final query holds the attributes used in final query. So only this particular set of
        # of attributes will be rewarded
        final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
        final_query = defaultdict()
        for every in final:
            final_query[every] = 1

        total = 0
        precision_at_k = 0
        no_of_intr = 0
        prev_interactions = queue.Queue(maxsize=2)
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            state = state.strip('[]')
            states = state.split(', ')

            interactions = []
            # Getting the attributes used for Testing on T4
            if task == 't3':
                if len(states) >= 1:
                    picked_attr = rbm.make_choice(k)
                    test = []
                    for s in states:
                        if len(s) >= 1:
                            interactions.append(s)
                            test.append(s)

                    if len(interactions) < 1:
                        continue

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    # Payoff is calculated based on the number of *correct attributes* in the current
                    # interaction
                    cnt = 0
                    for attrs in picked_attr:
                        if attrs in final_query:
                            cnt += 1

                    if cnt > 0:
                        rbm.update(list(prev_interactions.queue))

                    #Calculate Precision after 1st 3 interactions
                    if no_of_intr >= 3:
                        precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                        total += 1
                    no_of_intr += 1

        precision_at_k = precision_at_k / total
        return precision_at_k

    # def run_bush_mosteller_gt(self, k, alpha=0.2):
    #     rbm = BushMosteller(alpha, list(self.attributes.keys()))
    #     total = 0
    #     # k = 2  # K for Precision @ K
    #     precision_at_k = 0
    #     prev_interactions = queue.Queue(maxsize=2)
    #
    #     for row in self.data:
    #         userid, task, seqid, state = tuple(row)
    #         # Training on T2, T3, T4
    #         state = state.strip('[]')
    #         states = state.split(', ')
    #         interaction = []
    #         if task != 't1':
    #             for s in states:
    #                 if len(s) > 1 and s in self.attributes:
    #                     interaction.append(s)
    #                     # rae.update_qtable(1, s, 1, 0.4)
    #             if prev_interactions.full():
    #                 prev_interactions.get()
    #                 prev_interactions.put(interaction)
    #                 rbm.update(list(prev_interactions.queue))
    #
    #             else:
    #                 prev_interactions.put(interaction)
    #
    #     while not prev_interactions.empty():
    #         prev_interactions.get()
    #
    #     for row in self.cur_data:
    #         userid, task, seqid, state = tuple(row)
    #         # print("{} {}".format(task, state))
    #         # Training on task T2 and T3
    #         state = state.strip('[]')
    #         states = state.split(', ')
    #         # print(len(states))
    #
    #         interactions = []
    #         # if len(state) <= 3:
    #         #     continue
    #         if task != 't4' and task != 't1':
    #             for s in states:
    #                 if len(s) > 1 and s in self.attributes:
    #                     interactions.append(s)
    #                     # rae.update_qtable(user, s, 1, 0.2)
    #             if prev_interactions.full():
    #                 prev_interactions.get()
    #                 prev_interactions.put(interactions)
    #                 rbm.update(list(prev_interactions.queue))
    #             else:
    #                 prev_interactions.put(interactions)
    #
    #         # Getting the attributes used for Testing on T4
    #         elif task == 't4':
    #             if len(states) >= 1:
    #                 picked_attr = rbm.make_choice(k)
    #                 test = []
    #                 for s in states:
    #                     if len(s) >= 1 and s in self.attributes:
    #                         # rae.update_qtable(user, s, 2, 0.2)
    #                         interactions.append(s)
    #                         test.append(s)
    #                 # print("interactions {} {}".format(interactions, len(interactions)))
    #                 if len(interactions) < 1:
    #                     continue
    #
    #                 if prev_interactions.full():
    #                     prev_interactions.get()
    #                     prev_interactions.put(interactions)
    #                 else:
    #                     prev_interactions.put(interactions)
    #
    #                 rbm.update(list(prev_interactions.queue))
    #                 precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
    #                 total += 1
    #
    #     # rae.tester(user)
    #     precision_at_k = precision_at_k / total
    #     return precision_at_k

if __name__ == '__main__':
    a = TrainingBushMosteller()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")

    #Calculates average precision for individual user trained against all other users
    average_precision = 0
    epoch = 20
    k = 4
    for user in a.users:
        avrg_user = 0
        for experiment in range(epoch):
            a.read_cur_data(user)
            accu = a.run_bush_mosteller_v2(k, 0.2)
            avrg_user += accu
        average_precision += (avrg_user / epoch)
    average_precision /= len(a.users)
    print("P@{} for Bush and Mosteller Task 4 = {}".format(k, average_precision))

    k = 4
    average_precision = 0
    for user in a.users:
        avrg_user = 0
        for experiment in range(epoch):
            a.read_cur_data(user)
            accu = a.run_bush_mosteller_v1(k, 0.2)
            avrg_user += accu
        average_precision += (avrg_user / epoch)
    average_precision /= len(a.users)
    print("P@{} for Bush and Mosteller Task 3 = {}".format(k, average_precision))

    #Calculates average precision for individual user trained against all other users
    # average_precision = 0
    # for experiment in range(50):
    #     avrg_user = 0
    #     for user in a.users:
    #         a.read_data(user)
    #         a.read_cur_data(user)
    #         accu = a.run_bush_mosteller_gt(2, 0.2)
    #         avrg_user += accu
    #     average_precision += (avrg_user / len(a.users))
    # average_precision /= 50
    # print("P@k for Bush and Mosteller (Group)= {}".format(average_precision))
