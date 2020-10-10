import sqlite3
import re
import numpy as np
from collections import defaultdict
from modified_roth_and_erev import modified_roth_and_erev
import queue
import pdb

class conditional_dependency:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        self.users = [1,5,9,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        # self.users = [1, 5, 9, 109, 29, 57, 61, 81, 85, 97]
        self.attr_dict = dict()
        self.attributes = defaultdict()

    # Creating a connection with the database using sqlite3
    def create_connection(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)

    # Read all the information form the database and store important ones inside the class \ current user
    def read_data(self, userid):
        c = self.conn.cursor()
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid <> ' + str(
            userid)
        c.execute(query)
        self.data = c.fetchall()

    # Read all the information form the database and store important ones inside the class for current user
    def read_cur_data(self, userid):
        c = self.conn.cursor()
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'weather1\' and userid = ' + str(
            userid)
        c.execute(query)
        self.cur_data = c.fetchall()

    # For peeking into the saved data (esp. the attributes) as tuples
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
        for attr in self.attr_dict.copy():
            if self.attr_dict[attr] < 10 or len(attr) < 1:
                del self.attr_dict[attr]
            else:
                self.attributes[attr] = 1

    # Calculates the Mean Reciprocal Rank of predicted attributes against ground value
    def find_MRR(self, ground, test):
        mrr = 0
        for attr in ground:
            r = 0
            for attr_test in test:
                if attr == attr_test:
                    break
                r += 1
            if r != 0:
                mrr += 1 / r
        mrr /= len(ground)
        # print("Mean Reciprocal Rank ", end = " ")
        # print(mrr)
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

    # Running the Roth and Erev algorithm for an individual user (Version 2)
    # Tests on T4 and has online training, and Calculates Precision

    def run_roth_and_erev_v2(self, user):
        rae = modified_roth_and_erev()
        total = 0
        cnt = 0
        # print("Here")
        k = 4  # K for Precision @ K
        final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
        final_query = defaultdict()
        for every in final:
            final_query[every] = 1

        precision_at_k = 0
        prev_interactions = queue.Queue(maxsize=2)
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            # Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            interactions = []
            if len(state) <= 3:
                continue
            if task != 't4' and task != 't1' and task != 't3':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        interactions.append(s)
                        # rae.update_qtable(user, s, 1, 0.2)
                if prev_interactions.full():
                    prev_interactions.get()
                    prev_interactions.put(interactions)
                else:
                    prev_interactions.put(interactions)

                rae.update_qtable(user, list(prev_interactions.queue), 1, 0.2)

            # Getting the attributes used for Testing on T4
            if task == 't3':
                if len(states) >= 1:
                    picked_attr = rae.make_choice(user, list(prev_interactions.queue), k, 0.1)
                    test = []
                    for s in states:
                        if len(s) >= 1 and s in self.attributes:
                            # rae.update_qtable(user, s, 2, 0.2)
                            interactions.append(s)
                            test.append(s)

                    if len(interactions) < 1:
                        continue

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    cnt = 0
                    for attrs in picked_attr:
                        # if len(interactions )> 3:
                        #     pdb.set_trace()
                        if attrs in final_query:
                            cnt += 1
                    payoff = 1
                    # if cnt >= 2:
                    #     # print("Present")
                    #     payoff = 10
                    rae.update_qtable(user, list(prev_interactions.queue), payoff, 0.2)

                    # rae.update_qtable(user, list(prev_interactions.queue), 2, 0.2)
                    precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                    total += 1

        # rae.tester(user)
        precision_at_k = precision_at_k / total
        return precision_at_k

    # Training the Modified Roth and Erev algorithm on whole dataset [T2-T4 all users]
    # and testing on T4 of an individual user (Version 2)
    # Tests on T4 and has online training, and Calculates Precision

    def run_roth_and_erev_v4(self, user):
        rae = modified_roth_and_erev()
        prev_interactions = queue.Queue(maxsize=2)
        final_query = ["incident_date", "number of records", "precip", "sky"]

        for row in self.data:
            userid, task, seqid, state = tuple(row)
            # Training on T2, T3, T4
            state = state.strip('[]')
            states = state.split(', ')
            interaction = []
            if task == 't3':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        interaction.append(s)
                        # rae.update_qtable(1, s, 1, 0.4)
                if prev_interactions.full():
                    prev_interactions.get()
                    prev_interactions.put(interaction)
                else:
                    prev_interactions.put(interaction)

                cnt = 0
                for attrs in interaction:
                    if attrs in final_query:
                        cnt += 1
                payoff = 1
                if cnt >= 2:
                    payoff = 2
                rae.update_qtable(user, list(prev_interactions.queue), payoff, 0.2)

                # rae.update_qtable(user, list(prev_interactions.queue), 1, 0.2)

        while not prev_interactions.empty():
            prev_interactions.get()

        total = 0
        cnt = 0
        k = 4  # K for Precision @ K
        precision_at_k = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)
            # Training on task T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            interactions = []
            # if len(state) <= 3:
            #     continue
            # if task != 't4' and task != 't1':
            #     for s in states:
            #         if len(s) > 1 and s in self.attributes:
            #             interactions.append(s)
            #             # rae.update_qtable(user, s, 1, 0.2)
            #     if prev_interactions.full():
            #         prev_interactions.get()
            #         prev_interactions.put(interactions)
            #     else:
            #         prev_interactions.put(interactions)
            #
            #     rae.update_qtable(user, list(prev_interactions.queue), 1, 0.2)

            # Getting the attributes used for Testing on T4
            if task == 't3':
                if len(states) >= 1:
                    picked_attr = rae.make_choice(user, list(prev_interactions.queue), k, 0.1)
                    test = []
                    for s in states:
                        if len(s) >= 1 and s in self.attributes:
                            # rae.update_qtable(user, s, 2, 0.2)
                            interactions.append(s)
                            test.append(s)

                    if len(interactions) < 1:
                        continue

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    #Newly added
                    flag = 0
                    cnt = 0
                    for attrs in interactions:
                        if attrs in final_query:
                            cnt += 1
                    payoff = 1
                    if cnt >= 2:
                        payoff = 2
                    rae.update_qtable(user, list(prev_interactions.queue), payoff, 0.2)
                    precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                    total += 1

            # Used for printing the Queue
            # for items in list(prev_interactions.queue):
            #     print(items, end=" ")
            # print()
        # rae.tester(user)

        precision_at_k = precision_at_k / total
        # print("testing: {}".format(precision_at_k))

        return precision_at_k


if __name__ == '__main__':
    a = conditional_dependency()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")
    for user in a.users:
        a.read_cur_data(user)
        a.read_states(user)

    # removes attributes with minimum appearances
    a.remove_attr()
    print(a.attr_dict.keys())

    # # Calculates P@k for Modified Roth and Erev: Individual User
    # average_precision = 0
    # for experiment in range(50):
    #     avrg_user = 0
    #     for user in a.users:
    #         a.read_cur_data(user)
    #         accu = a.run_roth_and_erev_v2(user)
    #         # print("User: {} Precision@2 {}".format(user, accu))
    #         avrg_user += accu
    #         # break
    #     average_precision += (avrg_user / len(a.users))
    # average_precision /= 50
    # print("P@k for Roth and Erev = {}".format(average_precision))

    # Calculates P@k for Modified Roth and Erev: Trained over the whole group
    # average_precision = 0
    # for experiment in range(50):
    #     avrg_user = 0
    #     for user in a.users:
    #         a.read_data(user)
    #         a.read_cur_data(user)
    #         accu = a.run_roth_and_erev_v4(user)
    #         # print("User: {} Precision@2 {}".format(user, accu))
    #         avrg_user += accu
    #         # break
    #     average_precision += (avrg_user / len(a.users))
    # average_precision /= 50
    # print("P@k for Roth and Erev = {}".format(average_precision))
