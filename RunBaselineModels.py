import sqlite3
from collections import defaultdict
from LatestReward import LatestReward
import Categorizing
import queue
import pdb

#Main goal of this class is to find out which model replicates
#User Learning perfectly

#Training performed on individual user on Birdsstrikes1 dataset


class RunBaselineModels:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        # Birdstrikes1
        #Task 3
        # self.users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        #Task 4
        # self.users = [1,5,109,13,25,29,33,53,57,61,73,97]

        #weather1
        self.users = [1, 5, 21, 25, 29, 45, 53, 65, 69, 73, 77, 93, 97, 113, 117]
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
        query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'weather1\' and userid = ' + str(
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
        return precision_at_k

#Running the baseline of Latest-Reward
    def LatestReward(self, task, k):
        c = Categorizing.Categorizing()
        c.weather1()

        total = 0
        #Birdstrikes1
        # all_attrs = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"', '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"', '"operator"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"', '"type_eng"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"indicated_damage"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"reported_date"', '"warned"', '"dam_prop"', '"dam_rad"', '"index_nr"', '"speed"', '"incident_month"', '"faaregion"', '"location"', '"airport_id"']
        # all_attrs = c.get_category(all_attrs)
        # final = []
        # if task == 't3':
        #     final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
        # elif task == 't4':
        #     final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
        #              '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
        # final = c.get_category(final)

        #Weather1
        all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"', '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"', '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"', '"tornado"']
        all_attrs = c.get_category(all_attrs)
        final = []
        if task == 't3':
            final = ['"highwinds"', '"state"', '"lat"', '"lang"']
        elif task == 't4':
            final = ['"date"', '"lat"', '"lng"', '"state"',  '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
                     '"tmax"', '"tmin"', 'snow', 'fog']
        final = c.get_category(final)

        base = LatestReward(all_attrs, final, 1)
        # pdb.set_trace()

        precision_at_k = 0
        no_of_intr = 0
        cnt_attr = 0
        for row in self.cur_data:
            userid, ttask, seqid, state = tuple(row)
            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')
            #Removing add-worksheet (Null attribute) interaction
            if len(state) <= 3:
                continue

            if ttask == task:
                if len(states) >= 1:
                    picked_attr = base.make_choice(k)
                    test = []
                    for s in states:
                        test.append(s)
                    # print(test)
                    test = c.get_category(test)
                    cnt_attr += len(test)
                    base.assign_reward()
                    #Calculate precision after 1st 3 interactions
                    if no_of_intr >= 3:
                        precision_at_k += self.find_Precision_at_k(test, picked_attr, k)
                        total += 1
                    no_of_intr += 1
                    # print(test)
                    # print("picked {}".format(picked_attr))
        precision_at_k = precision_at_k / total
        # print("{:.2f},".format(cnt_attr / no_of_intr), end=' ')
        # c.show()
        return precision_at_k


if __name__ == '__main__':
    a = RunBaselineModels()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")

    # Calculates P@k for Modified Roth and Erev: Individual User
    average_precision = 0
    epoch = 20
    k = 5
    for user in a.users:
        avrg_user = 0
        a.read_cur_data(user)
        for experiment in range(epoch):
            accu = a.LatestReward('t4', k)
            avrg_user += accu
            # break
        average_precision += (avrg_user / epoch)
        # break
    average_precision /= len(a.users)
    print("P@{} for Roth and Erev = {}".format(k, average_precision))

#P@3 for Task 4 = 0.51 [birdstrikes1]

    # average_precision = 0
    # epoch = 20
    # k = 3
    # for user in a.users:
    #     avrg_user = 0
    #     a.read_cur_data(user)
    #     for experiment in range(epoch):
    #         accu = a.LatestReward('t3', k)
    #         avrg_user += accu
    #         # break
    #     average_precision += (avrg_user / epoch)
    #     # break
    # average_precision /= len(a.users)
    # print("P@{} for Roth and Erev = {}".format(k, average_precision))

#P@3 for Task 3 = 0.84 [birdstrikes1]
