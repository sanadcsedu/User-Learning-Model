import sqlite3
import re
import numpy as np
from collections import defaultdict
from roth_and_erev import roth_and_erev

class load_data:

    def __init__(self):
        self.conn = None
        self.data = None
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

    #Read all the information form the database and store important ones inside the class
    def read_data(self, userid):
        c = self.conn.cursor()
        # print(self.users)
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid = ' + str(userid)
        c.execute(query)
        # c.execute('select distinct userid from master_file where dataset = \'birdstrikes1\'')
        self.data = c.fetchall()
        # print(self.data)
        # for row in self.data:
        #     print(row[0], end=",")

    #For peeking into the saved data (esp. the attributes) as tuples
    def read_states(self, user):
        # print("########## " + str(user) + " #############")
        for row in self.data:
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
        print("Ground " + str(len(ground)))
        mrr /= len(ground)
        # print("Mean Reciprocal Rank ", end = " ")
        # print(mrr)
        return mrr

    #Running the Roth and Erev algorithm for an individual user
    def run_roth_and_erev(self, user):
        rae = roth_and_erev()
        for row in self.data:
            userid, task, seqid, state = tuple(row)
            #Training on task T1, T2 and T3
            state = state.strip('[]')
            states = state.split(', ')
            if task != 't4' and task != 't1':
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        rae.update_qtable(user, s, 1, 0.1)
            #Getting the attributes used for Testing on T4
            elif task == 't4':
                # picked_attr = rae.make_choice(user, 0.25)
                for s in states:
                    if len(s) > 1 and s in self.attributes:
                        self.t4attributes[s] = 1

        # print(self.t4attributes.keys())

        rae.update_prob_qtable(user)
        picked_attr = rae.make_choice(user, len(self.t4attributes), 0.1, self.attributes.copy())
        # picked_attr = rae.random_choice(user, len(self.t4attributes), self.attributes.copy())

        # print(picked_attr.keys())
        # print(rae.make_choice(user, 0.5))
        mrr = self.find_MRR(self.t4attributes, picked_attr)
        self.t4attributes.clear()
        return mrr

if __name__ == '__main__':
    a = load_data()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")
    for user in a.users:
        a.read_data(user)
        a.read_states(user)

    a.remove_attr()
    # print(len(a.attributes))
    # print("test")
    # print(a.attributes.keys())

    # a.read_data(57)
    # rr = a.run_roth_and_erev(57)
    # print(rr)
    # temp = defaultdict()
    mrr = 0
    for experiment in range(50):
        for user in a.users:
            print("User " + str(user))
            a.read_data(user)
            # print("Reciprocal Rank ", end=" ")
            rr = a.run_roth_and_erev(user)
            # temp[user] = rr
            print(rr)
            mrr += rr
        mrr /= len(a.users)

    print("MRR Random Choice ", end="")
    print(mrr)
    #Mean Reciprocal Rank using random choice 0.245
    #Mean Reciprocal Rank using Roth and Erev .33 (individual)


