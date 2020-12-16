import sqlite3
from collections import defaultdict
from LatestReward import LatestReward
from WinKeepLoseRandomize import WinKeepLoseRandomize
import Categorizing
import Evaluators
import queue
import pdb

#Main goal of this class is to find out which model replicates User Learning perfectly
#It tests heuristic models such as "Latest-Reward" and "Win-Keep Lose-Randomize"
#These models are relatively simple and popular in Game Theory and Behavioral Economics community

#Training performed on individual user on Birdsstrikes1, Weather1 Tableau dataset


class RunSimpleModels:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        # Birdstrikes1
        #Task 3
        self.users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
        #Task 4
        # self.users = [1,5,109,13,25,29,33,53,57,61,73,97]

        #weather1
        # self.users = [1, 5, 21, 25, 29, 45, 53, 65, 69, 73, 77, 93, 97, 113, 117]
        # self.users = [29]
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

#Running Latest-Reward
    def LatestReward(self, task, k):
        e = Evaluators.Evaluators()
        c = Categorizing.Categorizing()
        # c.weather1()
        c.birdstrikes1()

        total = 0
        #Birdstrikes1
        all_attrs = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"', '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"', '"operator"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"', '"type_eng"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"indicated_damage"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"reported_date"', '"warned"', '"dam_prop"', '"dam_rad"', '"index_nr"', '"speed"', '"incident_month"', '"faaregion"', '"location"', '"airport_id"']
        all_attrs = c.get_category(all_attrs)
        final = []
        if task == 't3':
            final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
            after = 1
        elif task == 't4':
            final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
                     '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
            after = 5
        final = c.get_category(final)

        #Weather1
        # all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"', '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"', '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"', '"tornado"']
        # all_attrs = c.get_category(all_attrs)
        # final = []
        # if task == 't3':
        #     final = ['"highwinds"', '"state"', '"lat"', '"lang"']
        #     after = 1
        # elif task == 't4':
        #     final = ['"date"', '"lat"', '"lng"', '"state"',  '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
        #              '"tmax"', '"tmin"', 'snow', 'fog']
        #     after = 5
        # final = c.get_category(final)

        base = LatestReward(all_attrs, final, 1)
        # pdb.set_trace()

        f1_score = 0
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
                    ground = []
                    for s in states:
                        ground.append(s)
                    # print(ground)
                    ground = c.get_category(ground)
                    cnt_attr += len(ground)
                    base.assign_reward()
                    #Calculate precision after 1st 3 interactions
                    if no_of_intr >= after:
                        if len(ground) == 0:
                            continue
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        f1_score += get_f1
                        total += 1
                    no_of_intr += 1
                    # print(ground)
                    # print("picked {}".format(picked_attr))
        f1_score = f1_score / total
        # print("{:.2f},".format(cnt_attr / no_of_intr), end=' ')
        # c.show()
        return f1_score

#Running Win-Keep Lose-Randomize
    def WinKeepLoseRandomize(self, task, k):
        e = Evaluators.Evaluators()
        c = Categorizing.Categorizing()
        c.birdstrikes1()

        total = 0
        # Birdstrikes1
        all_attrs = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"', '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"', '"operator"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"', '"type_eng"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"indicated_damage"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"reported_date"', '"warned"', '"dam_prop"', '"dam_rad"', '"index_nr"', '"speed"', '"incident_month"', '"faaregion"', '"location"', '"airport_id"']
        all_attrs = c.get_category(all_attrs)
        final = []
        if task == 't3':
            final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
            after = 3
        elif task == 't4':
            final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
                     '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
            after = 5
        final = c.get_category(final)

        # Weather1
        # all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"',
        #              '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"',
        #              '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"',
        #              '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"',
        #              '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"',
        #              '"thunder"', '"tornado"']
        # all_attrs = c.get_category(all_attrs)
        # final = []
        # if task == 't3':
        #     final = ['"highwinds"', '"state"', '"lat"', '"lang"']
        #     after = 1
        # elif task == 't4':
        #     final = ['"date"', '"lat"', '"lng"', '"state"', '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
        #              '"tmax"', '"tmin"', 'snow', 'fog']
        #     after = 5
        # final = c.get_category(final)

        sim = WinKeepLoseRandomize(all_attrs, final)
        # pdb.set_trace()

        f1_score = 0
        no_of_intr = 0
        cnt_attr = 0
        for row in self.cur_data:
            userid, ttask, seqid, state = tuple(row)
            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')
            # Removing add-worksheet (Null attribute) interaction
            if len(state) <= 3:
                continue

            if ttask == task:
                if len(states) >= 1:
                    picked_attr = sim.make_choice(k)
                    ground = []
                    for s in states:
                        ground.append(s)
                    ground = c.get_category(ground)
                    cnt_attr += len(ground)
                    # Calculate precision after 1st 3 interactions
                    if no_of_intr >= after:
                        if len(ground) == 0:
                            continue
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        f1_score += get_f1
                        total += 1
                    no_of_intr += 1
                    # print("test   {}".format(test))
                    # print("picked {}".format(picked_attr))
                    sim.assign_reward(picked_attr)
        # pdb.set_trace()
        f1_score = f1_score / total
        # print("User {}: f1 {}".format(user, f1_score))
        return f1_score

#Running a Simple Baseline
    def RunBaseline(self, task, k, cat = False):
        e = Evaluators.Evaluators()
        c = Categorizing.Categorizing()
        # c.weather1()
        c.birdstrikes1()
        total = 0
        # Birdstrikes1
        # all_attrs = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"', '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"', '"operator"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"', '"type_eng"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"indicated_damage"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"reported_date"', '"warned"', '"dam_prop"', '"dam_rad"', '"index_nr"', '"speed"', '"incident_month"', '"faaregion"', '"location"', '"airport_id"']
        # all_attrs = c.get_category(all_attrs)
        # final = []
        # if task == 't3':
        #     final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
        # elif task == 't4':
        #     final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
        #              '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
        # final = c.get_category(final)

        # Weather1
        # all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"',
        #              '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"',
        #              '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"',
        #              '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"',
        #              '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"',
        #              '"thunder"', '"tornado"']
        # all_attrs = c.get_category(all_attrs)
        # final = []
        # if task == 't3':
        #     final = ['"highwinds"', '"state"', '"lat"', '"lang"']
        #     after = 1
        # elif task == 't4':
        #     final = ['"date"', '"lat"', '"lng"', '"state"', '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
        #              '"tmax"', '"tmin"', 'snow', 'fog']
        #     after = 5
        # final = c.get_category(final)

        # sim = WinKeepLoseRandomize(all_attrs, final)
        # pdb.set_trace()

        f1_score = 0
        no_of_intr = 0
        cnt_attr = 0
        # picked_attr = ['"highwinds"', '"state"', '"longitude (generated)"', '"latitude (generated)"', '"tmin_f"', '"tmax_f"', '"date"']
        picked_attr = ['"incident_date"', '"precip"', '"sky"', '"ac_class"', '"damage"']
        if cat:
            picked_attr = c.get_category(picked_attr)
        for row in self.cur_data:
            userid, ttask, seqid, state = tuple(row)
            # Getting the states
            state = state.strip('[]')
            states = state.split(', ')
            # Removing add-worksheet (Null attribute) interaction
            if len(state) <= 3:
                continue

            if ttask == task:
                if len(states) >= 1:
                    ground = []
                    for s in states:
                        ground.append(s)
                    if cat:
                        ground = c.get_category(ground)
                    cnt_attr += len(ground)
                    # pdb.set_trace()
                    # Calculate precision after 1st 3 interactions
                    if no_of_intr >= 1:
                        if len(ground) == 0:
                            continue
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        f1_score += get_f1
                        total += 1
                    no_of_intr += 1
                    # print("test   {}".format(test))
                    # print("picked {}".format(picked_attr))
        # pdb.set_trace()
        f1_score = f1_score / total
        # print("User {}: f1 {}".format(user, no_of_intr))
        return f1_score


if __name__ == '__main__':
    a = RunSimpleModels()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")

    #Running a baseline
    f1_score = 0
    epoch = 20
    k = 4
    for user in a.users:
        avrg_user = 0
        a.read_cur_data(user)
        for experiment in range(epoch):
            accu = a.LatestReward('t3', 3)
            avrg_user += accu
            # break
        f1_score += (avrg_user / epoch)
        # break
    f1_score /= len(a.users)
    print("F-measure = {}".format(f1_score))

    # Calculates F-measure for Win-Keep Lose-Randomize
    # f1_score = 0
    # epoch = 20
    # k = 3
    # for user in a.users:
    #     avrg_user = 0
    #     a.read_cur_data(user)
    #     for experiment in range(epoch):
    #         accu = a.WinKeepLoseRandomize('t3', k)
    #         avrg_user += accu
    #         # break
    #     f1_score += (avrg_user / epoch)
    #     # break
    # f1_score /= len(a.users)
    # print("F-measure Win-Keep Lose-Randomize = {}".format(f1_score))

    # Calculates F-measure for Latest Reward
    # f1_score = 0
    # epoch = 20
    # k = 4
    # for user in a.users:
    #     avrg_user = 0
    #     a.read_cur_data(user)
    #     for experiment in range(epoch):
    #         accu = a.LatestReward('t4', k)
    #         avrg_user += accu
    #         # break
    #     f1_score += (avrg_user / epoch)
    #     # break
    # f1_score /= len(a.users)
    # print("F-measure Latest Reward = {}".format(f1_score))

#P@3 for Task 3 (Latest Reward) = 0.84 [birdstrikes1]
#P@3 for Task 3 (Win-Keep Lose-Randomize) = 0.37 [weather1]
