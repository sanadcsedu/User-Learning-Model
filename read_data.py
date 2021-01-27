import Categorizing
import sqlite3
import pdb

class read_data:

    def __init__(self):
        self.conn = None

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
        else: #faa1 dataset
            query = 'SELECT userid, task, seqId, state FROM master_file where dataset = \'faa1\' and userid = ' + str(userid)

        c.execute(query)
        cur_data = c.fetchall()
        return cur_data

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
            # priors = ['airplane', 'flight_related', 'weather', 'damaged_parts', 'event_time', 'aggregation', 'location', 'birds']
            priors = self.get_prior(dataset, task)
            users = [1, 5, 109, 13, 25, 29, 33, 53, 57, 61, 73, 97]
            if task == 't1':
                users = [1, 5, 109, 13, 25, 33, 53, 57, 61, 73]
                final = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"']
            elif task == 't2':
                users = [5, 109, 13, 25, 29, 33, 53, 57, 61, 73, 97]
                final = ['"ac_class"', '"damage"']
            elif task == 't3':
                # self.users = [1,5,109,13,25,29,33,37,53,57,61,73,77,81,85,97]
                final = ['"incident_date"', '"number of records"', '"precip"', '"sky"']
            else: #if task == 't4':
                final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"',
                         '"number of records"', '"precip"', '"sky"',
                         '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']
            if cat:
                all_attrs = c.get_category(all_attrs)
                final = c.get_category(final)
                priors = c.get_category(priors)

        elif dataset == 'weather1':
            c.weather1()
            # users = [1, 5, 21, 25, 29, 45, 53, 65, 69, 73, 77, 93, 97, 113, 117]
            users = [1, 5, 25, 29, 53, 65, 69, 73, 93, 97, 113, 117]

            all_attrs = ['"heavyfog"', '"number of records"', '"calculation(heavy fog (is null))"', '"date"', '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"', '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"', '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"', '"tornado"']
            # priors = ['fog', 'rain', 'snow', 'location', 'windy', 'time', 'aggregation', 'smoke', 'tornado', 'temperature']
            priors = self.get_prior(dataset, task)
            if task == 't1':
                users = [1, 5, 25, 53, 65, 69, 73, 93, 97, 113, 117]
                final = ['"heavyfog"', '"groundfog"', '"mist"', '"drizzle"']
            elif task == 't2':
                final = ['"tmax"', '"tmin"', '"date"']
            elif task == 't3':
                final = ['"highwinds"', '"state"', '"lat"', '"lng"']
                users = [1, 5, 25, 29, 53, 69, 73, 93, 97, 113, 117]
            else: # task == 't4':
                final = ['"date"', '"lat"', '"lng"', '"state"', '"tmax_f"', '"tmin_f"', '"prcp"', '"rain"',
                         '"tmax"', '"tmin"', '"snow"', '"fog"']
            if cat:
                all_attrs = c.get_category(all_attrs)
                final = c.get_category(final)
                priors = c.get_category(priors)

        else: #faa1
            c.faa1()
            all_attrs = ['"calculation(percent delta)"', '"destcityname"', '"calculation(arrival y/n)"', '"longitude (generated)"', '"deststate"', '"weatherdelay"', '"uniquecarrier"', '"crsdeptime"', '"deptime"', '"distance"', '"depdelay"', '"arrdelay"', '"calculation(delayed y/n)"', '"calculation(total delays)"', '"flightdate"', '"calculation(arrdelayed)"', '"carrierdelay"', '"calculation([arrdelay]+[depdelay])"', '"latitude (generated)"', '"airtime"', '"arrtime"', '"calculation(is delta flight)"', '"crselapsedtime"', '"taxiin"', '"crsarrtime"', '"originstate"', '"taxiout"', '"diverted"', '"lateaircraftdelay"', '"calculation(delay?)"', '"origincityname"', '"securitydelay"', '"cancellationcode"', '"origin"', '"calculation([dest]+[origin])"', '"nasdelay"', '"calculation(depdelayed)"', '"number of records"', '"cancelled"', '"dest"', '"actualelapsedtime"']
            users = [57, 45, 89, 65, 37, 93, 9, 33, 13, 21]
            # priors = ['time', 'carrier', 'diverted', 'origin', 'delay', 'aggregate', 'distance', 'cancellation', 'dest', 'taxi']
            priors = self.get_prior(dataset, task)
            if task == 't1':
                final = ['"depdelay"', '"arrdelay"', '"cancelled"', '"diverted"']
            elif task == 't2':
                final = ['"uniquecarrier"', '"flightdate"', '"number of records"']
                users = [57, 45, 89, 37, 93, 9, 33, 13, 21]
            elif task == 't3':
                final = ['"distance"', '"arrdelay"']
            else: # task == 't4':
                final = ['"uniquecarrier"', '"arrdelay"', '"depdelay"', '"origin"', '"dest"', '"number of records"']

            if cat:
                all_attrs = c.get_category(all_attrs)
                final = c.get_category(final)
                priors = c.get_category(priors)

        return users, all_attrs, priors, final

    #Priors for the learning algorithms
    #Assign some initial probabilities to the prior attributes to minimize cold start
    def get_prior(self, dataset, task):
        ret = None
        if dataset == 'birdstrikes1':
            if task == 't2':
                ret = ['"ac_class"', '"damage"']
            elif task == 't3':
                ret = ['"incident_date"', '"precip"', '"sky"']
            else: #task == 't4'
                ret = ['"damage"', '"precip"', '"sky"']
        if dataset == 'weather1':
            if task == 't2':
                ret = ['"tmax"', '"tmin"', '"date"']
            elif task == 't3':
                ret = ['"highwinds"', '"state"']
            else:  # task == 't4'
                ret = ['"tmax"', '"tmin"', '"heavyfog"', '"groundfog"', '"mist"', '"drizzle"']
        if dataset == 'faa1':
            if task == 't2':
                ret = ['"uniquecarrier"', '"flightdate"']
            elif task == 't3':
                ret = ['"distance"', '"arrdelay"']
            else:  # task == 't4'
                ret = ['"arrdelay"', '"depdelay"', '"cancelled"', '"diverted"', '"uniquecarrier"', '"flightdate"']
        return ret

if __name__ == '__main__':
    obj = read_data()
    obj.create_connection(r"D:\Tableau Learning\Tableau.db")
    datasets = ['birdstrikes1', 'weather1', 'faa1']
    tasks = ['t2', 't3', 't4']
    for d in datasets:
        for t in tasks:
            _, all, prior, final = obj.TableauDataset(d, t, True)