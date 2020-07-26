import sqlite3
import re
import numpy as np
from collections import defaultdict
from roth_and_erev import roth_and_erev

class group_training_v2:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        # self.users = [1,5,9,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        self.users = [1,5,9,109,29,57,61,81,85,97]
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
            # print(states)
            # print()

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

    #Calculates the Mean Reciprocal Rank of predicted attributes against ground value
    def find_MRR(self, ground, test):
        mrr = 0
        for attr in ground:
            r = 0
            for attr_test in test:
                if attr == attr_test:
                    break
                r += 1
            if r != 0:
                mrr += 1/r
        # print("Ground " + str(len(ground)))
        mrr /= len(ground)
        # print("Mean Reciprocal Rank ", end = " ")
        # print(mrr)
        return mrr

    #Running the Roth and Erev algorithm for an individual user (Version 1)
    #It doesn't update while using T4 and Calculates MRR
    def run_roth_and_erev_v1(self, user):
        rae = roth_and_erev()
        for row in self.data:
            userid, task, seqid, state = tuple(row)
            # Training on T2, T3, T4
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(1, s, 1, 0.4)

        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            #Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't4' and task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(1, s, 1, 0.2)
            #Getting the attributes used for Testing on T4
            elif task == 't4':
                # picked_attr = rae.make_choice(user, 0.25)
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        self.t4attributes[s] = 1

        # print(self.t4attributes.keys())
        # rae.tester(1)
        rae.update_prob_qtable(1)
        picked_attr = rae.make_choice(1, len(self.t4attributes), 0.1, self.attributes.copy())

        mrr = self.find_MRR(self.t4attributes, picked_attr)
        self.t4attributes.clear()
        return mrr

    # Calculates Precision@k of predicted attributes against ground value
    def find_Precision_at_k(self, ground, test, k):
        # print("Ground: ", end=" ")
        # print(ground)
        # print("Prediction: ", end=" ")
        # print(test)
        precision_at_k = 0
        found = 0
        for attr in ground:
            flag = 0
            for attr_test in test:
                if attr == attr_test:
                    flag = 1
            if flag == 1:
                found += 1
        precision_at_k = found / k
        # print("found {}".format(found))
        return precision_at_k

    #Running the Roth and Erev algorithm for an individual user (Version 2)
    #Tests on T4 and has online training, and Calculates Precision

    def run_roth_and_erev_v2(self, user):
        rae = roth_and_erev()
        # for row in self.data:
        #     userid, task, seqid, state = tuple(row)
        #     # Training on T2, T3, T4
        #     state = state.strip('[]')
        #     states = state.split(', ')
        #     if task != 't1':
        #         for s in states:
        #             if len(s) > 1 and s in self.attributes:
        #                 rae.update_qtable(1, s, 1, 0.4)

        total = 0
        cnt = 0
        k = 2 #K for Precision @ K
        precision_at_k = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            #Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't4' and task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(user, s, 1, 0.2)
            #Getting the attributes used for Testing on T4
            elif task == 't4':
                if len(states) >= 1:
                    picked_attr = rae.make_choice_v2(user, k, 0.25)
                    test = []
                    for s in states:
                        if len(s) >= 1 and s in self.attributes:
                            rae.update_qtable(user, s, 2, 0.2)
                            test.append(s)
                    precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                    total += 1

        precision_at_k = precision_at_k / total
        return precision_at_k

        # print(self.t4attributes.keys())
        # rae.tester(1)
        # rae.update_prob_qtable(1)
        # picked_attr = rae.make_choice(1, len(self.t4attributes), 0.1, self.attributes.copy())

        # mrr = self.find_MRR(self.t4attributes, picked_attr)
        # self.t4attributes.clear()
        # return mrr

        # Running the Roth and Erev algorithm trained on all user's data (Version 3)
        # Tests on T4 of the current user and has online training, and Calculates Precision

    def run_roth_and_erev_v3(self, user):
        rae = roth_and_erev()
        for row in self.data:
            userid, task, seqid, state = tuple(row)
            # Training on T2, T3, T4
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(user, s, 1, 0.4)

        total = 0
        cnt = 0
        k = 3  # K for Precision @ K
        precision_at_k = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            # Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't4' and task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(user, s, 1, 0.2)
            # Getting the attributes used for Testing on T4
            elif task == 't4':
                if len(states) >= 1:
                    picked_attr = rae.make_choice_v2(user, k, 0.25)
                    test = []
                    for s in states:
                        if len(s) >= 1 and s in self.attributes:
                            rae.update_qtable(user, s, 2, 0.2)
                            test.append(s)
                    precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                    total += 1

        precision_at_k = precision_at_k / total
        return precision_at_k


    #Returns Precision @ K when we randomly select strategies
    def run_random_choice(self, user):
        rae = roth_and_erev()
        total = 0
        cnt = 0
        k = 2 #K for Precision @ K
        precision_at_k = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            #Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't4' and task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(user, s, 1, 0.2)
            #Getting the attributes used for Testing on T4
            elif task == 't4':
                if len(states) >= 1:
                    picked_attr = rae.random_choice(user, k)
                    test = []
                    for s in states:
                        if len(s) >= 1 and s in self.attributes:
                            rae.update_qtable(user, s, 2, 0.2)
                            test.append(s)
                    precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                    total += 1

        precision_at_k = precision_at_k / total
        return precision_at_k


if __name__ == '__main__':
    a = group_training_v2()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")
    for user in a.users:
        a.read_cur_data(user)
        a.read_states(user)

    #removes attributes with minimum appearances
    a.remove_attr()

    #Calculates average precision for individual user trained against all other users
    average_precision = 0
    for experiment in range(50):
        avrg_user = 0
        for user in a.users:
            a.read_data(user)
            a.read_cur_data(user)
            accu = a.run_roth_and_erev_v3(user)
            # print("User: {} Precision@2 {}".format(user, accu))
            avrg_user += accu
        average_precision += (avrg_user / len(a.users))
    average_precision /= 50
    print("P@k for Roth and Erev = {}".format(average_precision))


    # Taken average for 50 iterations

    # Mean Reciprocal Rank using random choice 0.245

    # Mean Reciprocal Rank using Roth and Erev .330 (individual) [W/O Forgetting]
    # Mean Reciprocal Rank using Roth and Erev .335 (individual) [Forgetting = .1]
    # Mean Reciprocal Rank using Roth and Erev .344 (individual) [Forgetting = .25]

    # Mean Reciprocal Rank using Roth and Erev .324 (Group) [W/O Forgetting]
    # Mean Reciprocal Rank using Roth and Erev .342 (Group) [Forgetting = .25]
    # Mean Reciprocal Rank using Roth and Erev .35 (Group) [Training F = .5 Forgetting = .25]



