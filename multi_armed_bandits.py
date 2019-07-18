import random
import numpy as np
import os
from termcolor import colored

'''
defines constants for common usage in all algorithm thingies
K is the number of arms per multi-armed bandit
T is the number of pulls per trial
stdev is the standard deviation of all of the distributions
P is the probability vector for each of the arms
centers is the vector containing each of the centers of the probability distributions in the corresponding positions.
samplesize is the sample size used in each algorithm evaluation thing
'''
K = 10
T = 200
stdev = 25
centers = [random.randint(50, 100) for _ in range(K)]
P = [lambda c, s: np.random.normal(c, s) for _ in centers]
samplesize = 25000


# returns a number based on the distribution held in P[n]
def pull(n):
    return P[n](centers[n], stdev)


# evaluates alg, by executing it samplesize times, and returns the average.
def eval_alg(alg):
    total = 0
    for i in range(samplesize):
        total += alg()
    return total / samplesize


'''
compares alg1 and alg2 by taking the sum of their differences
returns a list with the following elements:
position 0: winning algorithm function reference
position 1: losing algorithm function reference
position 2: winning algorithm average
position 3: losing algorithm average
position 4: average absolute difference
position 5: percent difference
'''
def compare_alg(alg1, alg2):
    a1 = eval_alg(alg1)
    a2 = eval_alg(alg2)
    diff = a1 - a2
    return [
        alg1 if diff > 0 else alg2,
        alg1 if diff <= 0 else alg2,
        a1 if diff > 0 else a2,
        a1 if diff <= 0 else a2,
        abs(diff),
        100 * abs(diff) / ((a1 + a2) / (2))
    ]


# takes the array returned by compare_alg and prints it.
def print_results(result, sigfigs=3):
    print('Winner:', colored(result[0].__name__, 'cyan'))
    print('Loser:', colored(result[1].__name__, 'red'))
    print('Winner average', colored(round(result[2], sigfigs), 'cyan'))
    print('Loser average', colored(round(result[3], sigfigs), 'red'))
    print('Average absolute difference:', colored(round(result[4], sigfigs), 'magenta'))
    print('Percent difference:', colored(round(result[5], sigfigs), 'magenta'), end='%\n')


# algorithm 0
# pulls each arm the same number of times.
def equal_pull():
    total = 0
    for i in range(T):
        total += pull(i % 10)
    return total


# algorithm 1
# pulls each arm once, then pulls the arm that returned the highest value for the rest of the runs
def onepass():
    total = 0
    lastmax = 0
    maxpos = 0
    for i in range(K):
        a = pull(i)
        total += a
        if a > lastmax:
            lastmax = a
            maxpos = i
    for i in range(T - K):
        total += pull(maxpos)
    return total


# algorithm 2
# pulls each arm twice, then pulls the arm with the highest total value
def twopass():
    total = 0
    values = [0 for _ in range(K)]
    for i in range(2*K):
        values[i % K] += pull(i % K)
    max = 0
    for i in range(K):
        if values[i] > values[max]:
            max = i
    for i in range(T - 2*K):
        total += pull(max)
    total += sum(values)
    return total


result = compare_alg(onepass, twopass)
print_results(result)
os.system('say done')