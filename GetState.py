from collections import defaultdict
import pdb

class GetState:

    def __init__(self):
        self.all_attrs = None
        self.states = None
        self.check = set()

    def birdstrikes1(self):
        # self.all_attrs = [('"dam_eng1"', 'known'), ('"dam_eng2"', 'known') , ('"dam_windshld"', 'known'), ('"dam_wing_rot"', 'known'), ('"number of records"', 'known'), ('"damage"', 'known'),
        #              ('"ac_class"', 'known'), ('"incident_date"', 'known'), ('"precip"', 'known'), ('"sky"', 'known'), ('"phase_of_flt"', 'unk'), ('"operator"', 'unk'), ('"ac_mass"', 'unk'),
        #              ('"state"', 'unk'), ('"size"', 'unk'), ('"birds_struck"', 'unk'), ('"time_of_day"', 'unk'), ('"type_eng"', 'unk'), ('"birds_seen"', 'unk'), ('"distance"', 'unk'),
        #              ('"height"', 'unk'), ('"dam_eng3"', 'unk'), ('"indicated_damage"', 'unk'), ('"dam_tail"', 'unk'), ('"dam_nose"', 'unk'), ('"dam_lghts"', 'unk'),
        #              ('"dam_lg"', 'unk'), ('"dam_fuse"', 'unk'), ('"dam_eng4"', 'unk'), ('"dam_other"', 'unk'), ('"reported_date"', 'unk'), ('"warned"', 'unk'), ('"dam_prop"', 'unk'),
        #              ('"dam_rad"', 'unk'), ('"index_nr"', 'unk'), ('"speed"', 'unk'), ('"incident_month"', 'unk'), ('"faaregion"', 'unk'), ('"location"', 'unk'),
        #              ('"airport_id"', 'unk'), ('"atype"', 'unk'), ('"airport"', 'unk'), ('"incident_year"', 'unk'), ('"longitude (generated)"', 'unk'), ('"latitude (generated)"', 'unk'),
        #              ('"calculation(phase of flt dedup)"', 'unk'), ('"time"', 'unk')]

        # self.all_attrs = [('"dam_eng1"', 'boolean'), ('"dam_eng2"', 'boolean') , ('"dam_windshld"', 'boolean'), ('"dam_wing_rot"', 'boolean'), ('"number of records"', 'aggregate'), ('"damage"', 'categorical'),
        #              ('"ac_class"', 'categorical'), ('"incident_date"', 'time'), ('"precip"', 'categorical'), ('"sky"', 'categorical'), ('"phase_of_flt"', 'categorical'), ('"operator"', 'categorical'), ('"ac_mass"', 'categorical'),
        #              ('"state"', 'categorical'), ('"size"', 'categorical'), ('"birds_struck"', 'numeric'), ('"time_of_day"', 'time'), ('"type_eng"', 'categorical'), ('"birds_seen"', 'numeric'), ('"distance"', 'numeric'),
        #              ('"height"', 'numeric'), ('"dam_eng3"', 'boolean'), ('"indicated_damage"', 'boolean'), ('"dam_tail"', 'boolean'), ('"dam_nose"', 'boolean'), ('"dam_lghts"', 'boolean'),
        #              ('"dam_lg"', 'boolean'), ('"dam_fuse"', 'boolean'), ('"dam_eng4"', 'boolean'), ('"dam_other"', 'boolean'), ('"reported_date"', 'time'), ('"warned"', 'boolean'), ('"dam_prop"', 'boolean'),
        #              ('"dam_rad"', 'boolean'), ('"index_nr"', 'numeric'), ('"speed"', 'numeric'), ('"incident_month"', 'time'), ('"faaregion"', 'categorical'), ('"location"', 'categorical'),
        #              ('"airport_id"', 'categorical'), ('"atype"', 'categorical'), ('"airport"', 'categorical'), ('"incident_year"', 'time'), ('"longitude (generated)"', 'aggregate'), ('"latitude (generated)"', 'aggregate'),
        #              ('"calculation(phase of flt dedup)"', 'aggregate'), ('"time"', 'time')]

        self.states = defaultdict()
        test = set()
        # final = ['"atype"', '"ac_class"', '"type_eng"', '"time_of_day"', '"incident_date"', '"number of records"',
        #         '"precip"', '"sky"', '"birds_struck"', '"state"', '"size"', '"height"', '"distance"', '"phase_of_flt"']

        for attrs, category in self.all_attrs:
            self.states[attrs] = category
            test.add(category)

        # pdb.set_trace()
        print(test)

        # print(self.states)
        # for attrs in final:
        #     test.add(self.states[attrs])
        # test = list(test)
        # print(test)

        # for cat in test:
        #     print("{} = [".format(cat), end=" ")
        #     for attrs, category in self.all_attrs:
        #         if category == cat:
        #             print(attrs, end=", ")
        #     print(" ]")

    def weather1(self):
        self.all_attrs = [('"heavyfog"', 'fog'), ('"number of records"', 'aggregation'), ('"calculation(heavy fog (is null))"', 'fog'), ('"date"', 'time'), ('"tmax_f"', 'temperature'),
                          ('"tmin_f"', 'temperature'), ('"latitude (generated)"', 'location'), ('"longitude (generated)"', 'location'), ('"lat"', 'location'), ('"lng"', 'location'),
                          ('"state"', 'location'), ('"freezingrain"', 'rain'), ('"blowingsnow"', 'snow'), ('"blowingspray"', 'snow'), ('"drizzle"', 'rain'), ('"dust"', 'windy'),
                          ('"fog"', 'fog'), ('"mist"', 'fog'), ('"groundfog"', 'fog'), ('"freezingdrizzle"','rain'), ('"glaze"', 'snow'), ('"hail"', 'hail'), ('"highwinds"', 'windy'),
                          ('"icefog"', 'fog'), ('"icepellets"', 'snow'), ('"prcp"', 'precip'), ('"rain"', 'rain'), ('"smoke"', 'smoke'), ('"tmax"', 'temperature'), ('"tmin"', 'temperature'),
                          ('"snow"', 'snow'), ('"snowgeneral"', 'snow'), ('"snwd"', 'snow'), ('"thunder"', 'rain'), ('"tornado"', 'tornado')]

        self.states = defaultdict()
        test = set()
        for attrs, category in self.all_attrs:
            self.states[attrs] = category
            test.add(category)
        test = list(test)
        # print(test)


    def get_state(self, cur_attrs):
        ret = set()
        # pdb.set_trace()
        for attr in cur_attrs:
            if attr in self.states:
                ret.add(self.states[attr])
            # else:
            #     self.check.add(attr)
        ret = list(ret)
        # pdb.set_trace()
        return ret

    def update(self, cur_attrs):
        for attr in cur_attrs:
            if attr in self.states:
                self.states[attr] = 'known'

        # pdb.set_trace()

        # print(self.states)

    def show(self):
        print(self.check)

if __name__ == '__main__':
    c = GetState()
    c.birdstrikes1()
