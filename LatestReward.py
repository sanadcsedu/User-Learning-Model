from collections import defaultdict
import pdb
class LatestReward:

    def __init__(self, all_attrs, final, r):
        self.strategies = defaultdict(list, {key: 0 for key in all_attrs})
        self.rewarded_strategies = final
        self.picked_strategy = None
        self.reward = r

    def make_choice(self, k):
        self.picked_strategy = []
        for items in sorted(self.strategies.items(), key=lambda x: x[1], reverse=True):
            self.picked_strategy.append(items[0])
            k -= 1
            if k == 0:
                break
        return self.picked_strategy

    def assign_reward(self):
        denom = len(self.picked_strategy)
        cnt = 0
        for attr in self.picked_strategy:
            if attr in self.rewarded_strategies:
                self.strategies[attr] += self.reward / denom
                cnt += 1
        # pdb.set_trace()
        denom = len(self.strategies) - cnt
        for attr in self.strategies:
            if attr not in self.picked_strategy:
                self.strategies[attr] += self.reward / denom
        # pdb.set_trace()
