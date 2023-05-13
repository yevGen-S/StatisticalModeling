import random
import math


# uniform distribution
def rnuni(a, b):
    return random.random() * (b - a) + a


# normal distribution 1
def rnnrm1(m, sigma):
    return m + sigma * (sum([random.random() for _ in range(12)]) - 6)


# normal distribution 2
def rnnrm2(m, sigma):
    return m + sigma * math.sqrt(-2 * math.log(random.random(), math.e)) * math.cos(2 * math.pi * random.random())


# exponential distribution
def rnexp(beta):
    return -beta * math.log(random.random(), math.e)


# hi^2 distribution
def rnchis(n):
    return sum([rnnrm2(0, 1) ** 2 for _ in range(n)])


# student distribution
def rnstud(n):
    return rnnrm2(0, 1) / math.sqrt(rnchis(n) / n)


# erlang distribution
def rnerlang(beta, k):
    return sum([rnexp(beta) for _ in range(k)])
