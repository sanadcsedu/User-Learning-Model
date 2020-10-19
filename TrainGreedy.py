import sqlite3
import numpy as np
from collections import defaultdict
from EpsilonGreedy import EpsilonGreedy
import queue
import Categorizing
import Evaluators
import pdb


class TrainGreedy:

    def __init__(self):
        self.conn = None
        self.data = None
        self.cur_data = None
        self.users = None
        self.attr_dict = dict()
        self.attributes = defaultdict()

    # Creating a connection with the database using sqlite3
    def create_connection(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)

    # Read all the information form the database and store important ones inside the class for current user
    def read_cur_data(self, userid, dataset):
        c = self.conn.cursor()
        if dataset == 'birdstrikes1':
            query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'birdstrikes1\' and userid = ' + str(userid)
        elif dataset == 'weather1':
            query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'weather1\' and userid = ' + str(userid)

        c.execute(query)
        self.cur_data = c.fetchall()

    #Loads Specified Dataset from Tableau
    def TableauDataset(self, dataset, task, cat = True):
        c = Categorizing.Categorizing()
        if dataset == 'birdstrikes1':
            c.birdstrikes1()
            all_attrs = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"',
                         '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"',
                         '"operator"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"',
                         '"type_eng"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"indicated_damage"',
                         '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"',
                         '"dam_other"', '"reported_date"', '"warned"', '"dam_prop"', '"dam_rad"', '"index_nr"',
                         '"speed"', '"incident_month"', '"faaregion"', '"location"', '"airport_id"']
            priors = ['airplane', 'flight_related', 'weather', 'damaged_parts', 'event_time', 'aggregation',
                      'location', 'birds']
            if task == 't3' or task == 't2':
                self.users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
                final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
            if task == 't4':
                self.users = [1, 5, 109, 13, 25, 29, 33, 53, 57, 61, 73, 97]
                final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"',
                         '"number of records"', '"precip"', '"sky"',
                         '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
            if cat:
                all_attrs = c.get_category(all_attrs)
                final = c.get_category(final)

        elif dataset == 'weather1':
            c.weather1()
            # self.users = [1, 5, 21, 25, 29, 45, 53, 65, 69, 73, 77, 93, 97, 113, 117]
            all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"', '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"', '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"', '"tornado"']
            self.users = [1, 5, 25, 29, 53, 65, 69, 73, 77, 93, 97, 113, 117]
            priors = ['hail', 'fog', 'rain', 'snow', 'location', 'windy', 'time', 'aggregation', 'smoke', 'precip',
                      'tornado', 'temperature']
            if task == 't3' or task == 't2':
                final = ['"highwinds"', '"state"', '"lat"', '"lng"']
            if task == 't4':
                final = ['"date"', '"lat"', '"lng"', '"state"', '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
                         '"tmax"', '"tmin"', '"snow"', '"fog"']
            if cat:
                all_attrs = c.get_category(all_attrs)
                final = c.get_category(final)

        return all_attrs, priors, final


    def run_epsilon_greedy(self, user, dataset, t, k, epsilon, cat = False):
        e = Evaluators.Evaluators()
        if cat:
            c = Categorizing.Categorizing()
            if dataset == 'birdstrikes1':
                c.birdstrikes1()
            else:
                c.weather1()

        total = 0
        all_attr, priors, final = self.TableauDataset(dataset, t, True)
        after = 5
        epg = EpsilonGreedy(all_attr, epsilon)

        f1_score = 0
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

            if task == t:
                if len(states) >= 1:
                    picked_attr = epg.make_choice(k, list(prev_interactions.queue))
                    ground = []
                    for s in states:
                        if len(s) >= 1:
                            count_attributes += 1
                            interactions.append(s)
                            ground.append(s)

                    if cat:
                        ground = c.get_category(ground)
                        interactions = c.get_category(interactions)

                    if prev_interactions.full():
                        prev_interactions.get()
                        prev_interactions.put(interactions)
                    else:
                        prev_interactions.put(interactions)

                    #Payoff is calculated based on the number of *correct attributes* in the current interaction
                    payoff = 0
                    for attrs in picked_attr:
                        if attrs in final:
                            payoff += 1

                    if payoff > 0:
                        epg.update(list(prev_interactions.queue), payoff)

                    if no_of_intr >= after and len(ground) > 0:
                        _, _, get_f1 = e.f1_score(ground, picked_attr)
                        f1_score += get_f1
                        total += 1
                    no_of_intr += 1

        f1_score = f1_score / total
        return f1_score


if __name__ == '__main__':
    a = TrainGreedy()
    a.create_connection(r"D:\Tableau Learning\Tableau.db")
    # Calculates P@k for Modified Roth and Erev: Individual User
    f1_score = 0
    epoch = 20
    dataset = 'weather1'
    task = 't4'
    k = 4
    epsilon = 0.25
    _, _, _ = a.TableauDataset(dataset, task)
    for user in a.users:
        avrg_user = 0
        a.read_cur_data(user, dataset)
        for experiment in range(epoch):
            accu = a.run_epsilon_greedy(user, dataset, task, k, epsilon, True)
            avrg_user += accu
            # break
        f1_score += (avrg_user / epoch)
        # break
    f1_score /= len(a.users)
    print("F1 Score = {}".format(f1_score))
