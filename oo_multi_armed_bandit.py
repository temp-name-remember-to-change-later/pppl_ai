import numpy as np
import os

class Bandit:
    """
    Initializes a K-armed bandit.
    The bandit has a probability distribution vector used to determine the outputs of each of its arms.
    The centers of the arm-distributions are normally distributed with a center `c` and standard deviation `s`.
    The distributions themselves are normal distributions with centers as determined above and standard deviation `sd`

    Parameters:
    `k`  : The number of arms in the bandit
    `c`  : The center of the distribution of the arm-distribution centers
    `s`  : The standard deviation of the distribution of the arm-distribution centers
    `sd` : The standard deviation of each of the arm-distributions.
    """
    def __init__(self, k, c, s, sd):
        self.K = k
        self.C = np.random.normal(loc=c, scale=s, size=k)
        self.SD = sd

    # Returns a value based on the selected arm's probability distribution.
    def dispense(self, n):
        return np.random.normal(loc=self.C[n], scale=self.SD)

    # Returns the number of arms in the bandit
    def get_arms(self):
        return self.K


class Agent:
    """
    Initializes an Agent connected to a Bandit.

    Parameters:
    `bandit`     : A reference to the bandit that this agent is connected to
    `turn_limit` : The number of turns this Agent may take
    """
    def __init__(self, bandit, turn_limit):
        self.turn_limit = turn_limit
        self.turn = 0
        self.total = 0
        self.bandit = bandit
        self.memory = [[] for _ in range(bandit.get_arms())]

    # "Pulls" arm n of the Bandit, and increments turns taken.
    def pull(self, n):
        self.turn += 1
        reward = self.bandit.dispense(n)
        self.total += reward
        self.memory[n].append(reward)

    # Returns True if the Agent still has turns to play; False otherwise
    def has_turns(self):
        return True if self.turn < self.turn_limit else False

    # Returns total award obtained
    def balance(self):
        return self.total

    # Returns the arm within the Agent's memory that has, on average, given the largest reward
    def max_in_mem(self):
        avg_list = [np.mean(arr) for arr in self.memory]
        return avg_list.index(max(avg_list))

    # Resets the bandit, setting turns, total, and memory back to their initial state.
    def rese(self):
        self.turn = 0
        self.total = 0
        self.memory = [[] for _ in range(self.bandit.get_arms())]

    # Algorithm 0; pull all arms equally.
    def pull_equally(self):
        while self.has_turns():
            self.pull(self.turn % self.bandit.get_arms())

    # Algorithm 1; pull all arms once, then continually pull the arm that returned the largest result.
    def pass_once(self):
        # `while... [] in self.memory` means "while self.memory contains unpulled arms"
        while self.has_turns() and [] in self.memory:
            self.pull(self.turn)
        while self.has_turns():
            self.pull(self.max_in_mem())

    # Algorithm 2; pull all arms n times, then continually pull the arm that returned the largest average.
    def pass_n_times(self, n):
        lengths = [len(arr) for arr in self.memory]
        while self.has_turns() and min(lengths) < n:
            lengths = [len(arr) for arr in self.memory]
            self.pull(self.turn % self.bandit.get_arms())
        while self.has_turns():
            self.pull(self.max_in_mem())


b = Bandit(10, 50, 25, 10)
a = Agent(b, 50)
PE_total = 0
PO_total = 0
PT_total = 0
for i in range(10000):
    a.pull_equally()
    PE_total += a.balance()
    a.reset()

    a.pass_once()
    PO_total += a.balance()
    a.reset()

    a.pass_n_times(2)
    PT_total += a.balance()
    a.reset()

print(PE_total)
print(PO_total)
print(PT_total)
os.system('say finished')