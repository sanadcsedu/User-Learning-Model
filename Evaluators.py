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

    def before_after(self, f1_score, total, threshold):
        # Check the f1 accuracy before/after certain threshold
        threshold = int(total * threshold)
        f1_before = f1_after = final_f1 = 0
        cnt1 = cnt2 = 0
        for idx in range(len(f1_score)):
            if idx < threshold:
                f1_before += f1_score[idx]
                cnt1 += 1
            else:
                f1_after += f1_score[idx]
                cnt2 += 1
            final_f1 += f1_score[idx]
        f1_before /= threshold
        f1_after /= (len(f1_score) - threshold)
        final_f1 /= len(f1_score)
        # pdb.set_trace()
        return f1_before, f1_after, final_f1


if __name__ == "__main__":
    e = Evaluators()
    p, r, f1 = e.f1_score(['A', 'B', 'C'], ['E', 'C'])
    print("{:.2f} {:.2f} {:.2f}".format(p, r, f1))
    p, r, f1 = e.f1_score(['A', 'B', 'C'], ['B', 'E', 'C', 'D'])
    print("{:.2f} {:.2f} {:.2f}".format(p, r, f1))