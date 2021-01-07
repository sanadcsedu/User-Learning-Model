from scipy.stats import wilcoxon
import numpy as np

class wilcoxon_test:
    # def __init__(self):

    def find_wilcoxon(self, data1, data2, algo1, algo2):
        # compare samples
        stat, p = wilcoxon(data1, data2)
        print("Comparison between {} Vs. {}".format(algo1, algo2))
        print('Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')
        print("Difference between Mean {:.2f}\n\n".format(np.mean(data1) - np.mean(data2)))

if __name__ == '__main__':
    # data laid out = [Weather1 T2, T3, T4, FAA1 T2, T3, T4]
    # roth = [0.67, 0.63, 0.63, 0.82, 0.64, 0.81]
    # bush = [0.67, 0.64, 0.64, 0.83, 0.69, 0.83]
    # epsilon = [0.7, 0.65, 0.64, 0.85, 0.72, 0.79]
    # adaptive = [0.7, 0.65, 0.64, 0.83, 0.7, 0.79]

    # #data laid out = [Birdstrike1 T2, T3, T4, Weather1 T2, T3, T4, FAA1 T2, T3, T4]
    roth = [0.8, 0.84, 0.77, 0.67, 0.63, 0.63, 0.82, 0.64, 0.81]
    bush = [0.80, 0.85, 0.78, 0.67, 0.64, 0.64, 0.83, 0.69, 0.83]
    epsilon = [0.8, 0.83, 0.73, 0.7, 0.65, 0.64, 0.85, 0.72, 0.79]
    adaptive = [0.89, 0.92, 0.79, 0.7, 0.65, 0.64, 0.83, 0.7, 0.79]

    obj = wilcoxon_test()
    obj.find_wilcoxon(roth, bush, 'roth', 'bush')
    obj.find_wilcoxon(roth, epsilon, 'roth', 'epsilon')
    obj.find_wilcoxon(roth, adaptive, 'roth', 'adaptive')
    obj.find_wilcoxon(bush, epsilon, 'bush', 'epsilon')
    obj.find_wilcoxon(bush, adaptive, 'bush', 'adaptive')
    obj.find_wilcoxon(epsilon, adaptive, 'epsilon', 'adaptive')





