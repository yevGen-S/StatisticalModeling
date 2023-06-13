import random
import math
from numpy import argmin

EPS = 0.0001
T_A = 3.0902
P0 = 0.999

NUMBER_OF_PARTS_TYPES = 3
n = int((T_A ** 2) * P0 * (1 - P0) / (EPS ** 2))
MAX_REPAIR_PARTS = 10

parts_amount_schema = [4, 2, 6]
lambda_ = [40e-6, 10e-6, 80e-6]


def prob_of_uptime(required_time, repair_parts):
    failures_amount = 0
    for _ in range(n):
        time = []
        for i in range(NUMBER_OF_PARTS_TYPES):
            type_time = []
            for __ in range(parts_amount_schema[i]):
                type_time.append(-1 * math.log(random.random()) / lambda_[i])
            for __ in range(repair_parts[i]):
                min_time_index = argmin(type_time)
                type_time[min_time_index] = type_time[min_time_index] - math.log(random.random()) / lambda_[i]
            for j in range(parts_amount_schema[i]):
                time.append(type_time[j])
        if not logic_func_of_sys_efficiency(time, required_time):
            failures_amount += 1
    return 1 - failures_amount / n


def logic_func_of_sys_efficiency(time, required_time):
    return ((time[0] > required_time and time[1] > required_time or time[2] > required_time and time[3] > required_time) and
            (time[4] > required_time and time[5] > required_time) and
            (time[6] > required_time and time[7] > required_time or time[8] > required_time and time[9] > required_time or
             time[10] > required_time and time[11] > required_time))


if __name__ == '__main__':
    needed_time = 8760
    repair_parts = [0] * NUMBER_OF_PARTS_TYPES

    for i in range(MAX_REPAIR_PARTS):
        repair_parts[0] = i
        for j in range(MAX_REPAIR_PARTS):
            repair_parts[1] = j
            for k in range(MAX_REPAIR_PARTS):
                repair_parts[2] = k
                p_value = prob_of_uptime(needed_time, repair_parts)
                if p_value > P0:
                    print(repair_parts, f"P = {p_value}, n = {sum(repair_parts)}")
