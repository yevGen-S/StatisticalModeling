import math
import random


# uniform distribution
def irnuni(r_low, r_up) -> int:
    return math.floor(r_low + random.random() * (r_up - r_low + 1))


# binomial_distribution
def irnbin(n, p) -> int:
    if n >= 100:
        return round(random.normalvariate(n * p, math.sqrt(n * p * (1 - p))) + 0.5)
    else:
        a = random.random()
        p_k = (1 - p) ** n
        m = 0

        while a >= p_k:
            a -= p_k
            p_k *= p * (n - m) / ((m + 1) * (1 - p))
            m += 1

        return m


# geom_distribution_1
def irngeo_1(p) -> int:
    a = random.random()
    p_k = p
    m = 0

    while a >= p_k:
        a -= p_k
        p_k *= (1 - p)
        m += 1

    return m


# geom_distribution_2
def irngeo_2(p):
    a = random.random()
    m = 0

    while a > p:
        a = random.random()
        m += 1

    return m


# geom_distribution_3
def irngeo_3(p):
    return round(math.log(random.random(), math.e) / math.log(1 - p, math.e))


# poisson_distribution_1
def irnpoi(mu):
    if mu >= 88:
        return random.normalvariate(mu, mu)
    else:
        a = random.random()
        p_k = math.exp(-mu)
        m = 1

        while a >= p_k:
            a -= p_k
            p_k *= mu / m
            m += 1

        return m


# poisson_distribution_2
def irnpsn(mu):
    if mu >= 88:
        return random.normalvariate(mu, mu)
    else:
        p_k = random.random()
        m = 1

        while p_k >= math.exp(-mu):
            p_k *= random.random()
            m += 1

        return m
