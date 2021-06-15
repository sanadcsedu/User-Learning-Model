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
        alpha = 0.2
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
    roth = [0.84, 0.80, 0.74, 0.88, 0.85, 0.83, 0.85, 0.52, 0.71]
    bush = [0.95, 0.85, 0.77, 0.82, 0.90, 0.85, 0.85, 0.57, 0.70]
    epsilon = [0.76, 0.82, 0.69, 0.82, 0.88, 0.82, 0.73, 0.59, 0.77]
    adaptive = [0.79, 0.83, 0.71, 0.83, 0.87, 0.84, 0.77, 0.65, 0.79]
    no_exploration = [0.77, 0.86, 0.66, 0.95,  0.85, 0.80, 0.78, 0.46, 0.68]

    obj = wilcoxon_test()
    obj.find_wilcoxon(roth, bush, 'roth', 'bush')
    obj.find_wilcoxon(roth, epsilon, 'roth', 'epsilon')
    obj.find_wilcoxon(roth, adaptive, 'roth', 'adaptive')
    obj.find_wilcoxon(bush, epsilon, 'bush', 'epsilon')
    obj.find_wilcoxon(bush, adaptive, 'bush', 'adaptive')
    obj.find_wilcoxon(epsilon, adaptive, 'epsilon', 'adaptive')
    obj.find_wilcoxon(roth, no_exploration, 'roth', 'no-exploration')
    obj.find_wilcoxon(bush, no_exploration, 'bush', 'no-exploration')
    obj.find_wilcoxon(epsilon, no_exploration, 'epsilon', 'no-exploration')
    obj.find_wilcoxon(adaptive, no_exploration, 'adaptive', 'no-exploration')

#mean values of different exploration algorithms
#Roth and Erev = 0.71
#Bush and Mosteller = 0.77
#Epsilon-Greedy = 0.64
#Adaptive Epsilon-Greedy = 0.66