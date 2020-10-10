import sqlite3
import re
import numpy as np
from collections import defaultdict
from modified_roth_and_erev import modified_roth_and_erev
import queue
import Categorizing
import pdb

#Main goal of this class is to find out which model replicates
#User Learning perfectly

#Training performed on individual user on Birdsstrikes1 dataset


class UserLearningV2:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        # Birdstrikes1
        # Task 3
        self.users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        # Task 4
        # self.users = [1,5,109,13,25,29,33,53,57,61,73,97]

        # weather1
        # self.users = [1, 5, 21, 25, 29, 45, 53, 65, 69, 73, 77, 93, 97, 113, 117]

        self.attr_dict = dict()
        self.attributes = defaultdict()

    # Creating a connection with the database using sqlite3
    def create_connection(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)

    # Read all the information form the database and store important ones inside the class for current user
    def read_cur_data(self, userid):
        c = self.conn.cursor()
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid = ' + str(
            userid)
        c.execute(query)
        self.cur_data = c.fetchall()

#In this experiment we are keeping all the attributes.
#In previous experiment attributes with less minimal frequency were not considered.

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
        # print("found {}".format(found))
        return precision_at_k

#Running Roth and Erev algorithm for an individual user (Task 3)
    def run_roth_and_erev(self, user, cat = False):
        if cat:
            c = Categorizing.Categorizing()
            c.birdstrikes1()
        rae = modified_roth_and_erev()
        total = 0

        #Setting up some prior strategies
        # priors = ['"incident_date"', '"precip"', '"sky"']
        priors = ['"longitude (generated)"', '"dam_tail"', '"faaregion"', '"indicated_damage"', '"index_nr"', '"sky"', '"latitude (generated)"', '"time_of_day"', '"reported_date"', '"damage"', '"birds_struck"', '"ac_class"', '"phase_of_flt"', '"distance"', '"dam_eng1"', '"height"', '"atype"', '"dam_lg"', '"birds_seen"', '"dam_wing_rot"', '"number of records"', '"state"', '"dam_windshld"', '"time"', '"location"', '"ac_mass"', '"dam_prop"', '"airport"', '"dam_fuse"', '"type_eng"', '"precip"', '"incident_year"', '"dam_nose"', '"calculation(phase of flt dedup)"', '"dam_other"', '"dam_rad"', '"dam_eng3"', '"warned"', '"dam_eng4"', '"dam_eng2"', '"incident_date"', '"dam_lghts"', '"size"', '"airport_id"', '"speed"', '"incident_month"', '"operator"']

        # if cat:
        #     priors = c.get_category(priors)
        # priors = ['airplane', 'flight_related', 'weather', 'damaged_parts', 'event_time', 'aggregation', 'location', 'birds']
        rae.add_prior_strategies(user, priors, 1)

        #Final query holds the attributes used in final query. So only this particular set of
        #of attributes will be rewarded
        final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
        if cat:
            final = c.get_category(final)
        final_query = defaultdict()
        for every in final:
            final_query[every] = 1

        k = 3  # K for Precision @ K
        precision_at_k = 0
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0
        count_attributes = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)

            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')
            interactions = []

            #Removing add-worksheet (Null attribute) interaction
            if len(state) <= 3:
                continue

            #Training on T2 (Skipped for T3 Training)
            # if task != 't4' and task != 't1' and task != 't3':
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

            # Getting the attributes used for Testing on T3
            # Training and Testing on T3
            if task == 't3':
                if len(states) >= 1:
                    picked_attr = rae.make_choice(user, list(prev_interactions.queue), k, 0.1)
                    test = []
                    for s in states:
                        if len(s) >= 1:
                            count_attributes += 1
                            interactions.append(s)
                            test.append(s)

                    if cat:
                        test = c.get_category(test)
                        interactions = c.get_category(interactions)

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    #Payoff is calculated based on the number of *correct attributes* in the current interaction
                    cnt = 0
                    for attrs in picked_attr:
                        if attrs in final_query:
                            cnt += 1
                    payoff = cnt

                    if payoff > 0:
                        rae.update_qtable(user, list(prev_interactions.queue), payoff, 0.1)

                    # rae.update_qtable(user, list(prev_interactions.queue), 2, 0.2)

                    #Calculate precision after 1st 3 interactions
                    if no_of_intr >= 3:
                        # print("TEST")
                        # print(test)
                        # print(picked_attr)
                        precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                        total += 1
                    no_of_intr += 1

        # rae.tester(user)
        # print("{:.2f},".format(count_attributes/no_of_intr))
        precision_at_k = precision_at_k / total
        return precision_at_k

#Running Roth and Erev algorithm for an individual user (Task 4)
    def run_roth_and_erev_v2(self, user, cat = False):
        if cat:
            c = Categorizing.Categorizing()
            c.weather1()

        rae = modified_roth_and_erev()
        total = 0

        #Setting up some prior strategies based on Task 2 and Task 3
        # priors = ['"incident_date"', '"precip"', '"sky"', '"birds_struck"', '"ac_class"']
        # priors = ['"highwinds"', '"state"', '"lat"', '"lang"', '"tmax"', '"tmin"', '"fog"'
        #           '"drizzle"', '"mist"', '"date"']
        # if cat:
        #     priors = c.get_category(priors)
        priors = ['hail', 'fog', 'rain', 'snow', 'location', 'windy', 'time', 'aggregation', 'smoke', 'precip', 'tornado', 'temperature']
        rae.add_prior_strategies(user, priors, 1)

        #Final query holds the attributes used in final query. So only this particular set of
        #of attributes will be rewarded
        # final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"', '"precip"', '"sky"',
        #          '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
        final = ['"date"', '"lat"', '"lng"', '"state"', '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
                 '"tmax"', '"tmin"', '"snow"', '"fog"']
        if cat:
            final = c.get_category(final)
        final_query = defaultdict()
        for every in final:
            final_query[every] = 1
        # pdb.set_trace()

        k = 4 # K for Precision @ K
        precision_at_k = 0
        prev_interactions = queue.Queue(maxsize=2)
        no_of_intr = 0
        count_attributes = 0
        for row in self.cur_data:
            userid, task, seqid, state = tuple(row)

            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')
            interactions = []

            #Removing add-worksheet (Null attribute) interaction
            if len(state) <= 3:
                continue

            # Getting the attributes used for Testing on T3
            # Training and Testing on T3
            if task == 't4':
                if len(states) >= 1:
                    picked_attr = rae.make_choice(user, list(prev_interactions.queue), k, 0.1)
                    test = []
                    for s in states:
                        if len(s) >= 1:
                            # count_attributes += 1
                            interactions.append(s)
                            test.append(s)

                    if cat:
                        interactions = c.get_category(interactions)
                        test = c.get_category(test)
                    count_attributes += len(test)

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    #Payoff is calculated based on the number of *correct attributes* in the current
                    #interaction
                    cnt = 0
                    for attrs in picked_attr:
                        if attrs in final_query:
                            cnt += 1
                    payoff = cnt

                    if payoff > 0:
                        rae.update_qtable(user, list(prev_interactions.queue), payoff, 0.1)

                    # rae.update_qtable(user, list(prev_interactions.queue), 2, 0.2)

                    #Calculate precision after 1st 3 interactions
                    if no_of_intr >= 5:
                        # print(user)
                        # print("Current interaction: ", end='  ')
                        # print(test)
                        # print("Predicted interaction: ", end='')
                        # print(picked_attr)
                        precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                        total += 1
                    no_of_intr += 1

        # rae.tester(user)
        # print("{:.2f},".format(count_attributes/no_of_intr), end=' ')
        precision_at_k = precision_at_k / total
        return precision_at_k


if __name__ == '__main__':
    a = UserLearningV2()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")

    # Calculates P@k for Modified Roth and Erev: Individual User
    average_precision = 0
    epoch = 20
    for user in a.users:
        avrg_user = 0
        a.read_cur_data(user)
        for experiment in range(epoch):
            accu = a.run_roth_and_erev(user, False)
            # print("User: {} Precision@K {}".format(user, accu))
            avrg_user += accu
            # break
        average_precision += (avrg_user / epoch)
        # break
    average_precision /= len(a.users)
    print("P@k for Roth and Erev = {}".format(average_precision))

#Task 3
#Average number of interactions for all the users: 2.67, So we are calculating P@3
#users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
#frequency of interactions = [2.2, 2.83, 3.49, 2.61, 3.1, 3.06, 2.12, 2.9, 3.33, 2.62, 2.65, 2.82, 1.53, 3.33, 1.55, 2.59]

#Task 4
#We are calculating p@k, where k = 4, because average #interactions/user = 2.98
#users self.users = [1,5,109,13,25,29,33,53,57,61,73,97]
#frequency of interactions = [2.13, 4.54, 3.56, 1.67, 2.81, 3.38, 1.97, 2.78, 3.83, 2.5, 3.43, 3.2]