class Evaluators:
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

    def precision(self, ground, test):
        tp = 0
        for attr in test:
            if attr in ground:
                tp += 1
        # fp + tp = len(test) total number of positive predictions made
        precision = tp / len(test)
        return precision

    def recall(self, ground, test):
        tp = 0
        for attr in ground:
            if attr in test:
                tp += 1
        #Recall is going to be the proportion of relevant strategies that has been
        #retrieved . R = (RET \ REL) / REL
        recall = tp / len(ground)
        return recall

    def f1_score(self, ground, test):
        p = self.precision(ground, test)
        r = self.recall(ground, test)
        if p + r == 0:
            f1 = 0
        else:
            f1 = (2 * p * r) / (p + r)
        return p, r, f1


if __name__ == "__main__":
    e = Evaluators()
    p, r, f1 = e.f1_score(['A', 'B', 'C', 'D', 'E'], ['E', 'C'])
    print("{} {} {}".format(p, r, f1))