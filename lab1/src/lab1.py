import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
import os

absolute_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])

relative_path_graph = "results/graphics"
relative_path_res = "results"

full_path_graphics = os.path.join(absolute_path, relative_path_graph)
full_path_res = os.path.join(absolute_path, relative_path_res)


def get_pseudo_rand_sequence(n):
    return [random.random() for _ in range(n)]


def get_empirical_expected_value(sequence):
    return sum(sequence) / len(sequence)


def get_empirical_variance(sequence, expected_value):
    return sum([(n - expected_value) ** 2 for n in sequence]) / len(sequence)


def get_empirical_square_deviation(empirical_variance):
    return math.sqrt(empirical_variance)


# Auto-correlation function to assess the degree of connectedness
# pseudo-random numbers
def K(f_shift, rand_sequence, expected_value):
    n = len(rand_sequence)
    return sum(
        [(rand_sequence[i] - expected_value) * (rand_sequence[i + f_shift] - expected_value) for i in
         range(n - f_shift)]) / \
           sum([(rand_sequence[i] - expected_value) ** 2 for i in range(n)])


def get_values_of_K_func_for_different_shifts(f_shift_size):
    pr_sequence = get_pseudo_rand_sequence(f_shift_size)
    expected_value = get_empirical_expected_value(pr_sequence)
    auto_correlation_func_values = [K(f, pr_sequence, expected_value) for f in range(f_shift_size)]
    return auto_correlation_func_values


def get_comparison_table():
    try:
        f = open(
            f"{full_path_res}/table-task-2.csv", "x")
    except:
        print("File is existing. Write to it")
        f = open(f"{full_path_res}/table-task-2.csv", "w")

    headers = ['n', 'M', 'teoretical M', 'M diff', 'D', 'teoretical D', 'D diff']
    writer = csv.writer(f, delimiter=',')
    writer.writerow(headers)

    teoretical_m = 0.5
    teoretical_d = 0.08333

    for n in [10, 100, 1000, 10000]:
        pr_sequence = get_pseudo_rand_sequence(n)
        expected_value = get_empirical_expected_value(pr_sequence)
        diff_teor_and_empir_values = abs(teoretical_m - expected_value)
        variance = get_empirical_variance(pr_sequence, expected_value)
        variance_diff = abs(teoretical_d - variance)
        print(
            f'n = {n}: \t @M@ = {expected_value}  \t teoretical M = {teoretical_m} \t diff = {diff_teor_and_empir_values} '
            f' \t @D@ = {variance} \t teoretical D = {teoretical_d} \t diff = {variance_diff}')
        writer.writerow(
            [n, expected_value, teoretical_m, diff_teor_and_empir_values, variance, teoretical_d, variance_diff])

    f.close()


def plot_correlograms():
    for f in [10, 100, 1000, 10000]:
        x_values = [i + 1 for i in range(f)]
        correlation_values = get_values_of_K_func_for_different_shifts(f)
        plt.figure()
        plt.xlabel('f')
        plt.ylabel('K(f)')
        plt.bar(x_values, correlation_values, edgecolor='black')
        print('done', f)
        plt.savefig(f'{full_path_graphics}/graphic-{f}.png')

    plt.show()


def plot_distribution_density(n):
    sequence = get_pseudo_rand_sequence(n)
    gap_values = {10: 10, 100: 10, 1000: 20, 10000: 20}
    gaps = gap_values[n]

    x_theory = np.linspace(0, 1, n)
    y_theory = [1] * n

    gap_width = 1 / gaps
    bins = [i * gap_width for i in range(gaps + 1)]

    plt.figure()
    plt.hist(sequence, bins=bins, density=True, edgecolor='black')  #
    plt.plot(x_theory, y_theory, color='red')
    plt.legend(['Теоретическая', 'Эмпирическая'])
    plt.suptitle(f'Функция плотности распределения \nn = {n}')
    plt.savefig(f'{full_path_graphics}/distribution_density-graphic-{n}.png')
    plt.show()


def plot_distribution(n):
    sequence = get_pseudo_rand_sequence(n)
    gap_values = {10: 10, 100: 10, 1000: 100, 10000: 1000}

    gaps = gap_values[n]
    gap_width = 1 / gaps

    x_theory = np.linspace(0, 1, n)
    y_theory = x_theory

    count, bins_count = np.histogram(sequence, bins=gaps)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    plt.figure()
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.plot(x_theory, y_theory, '--', color='red')

    plt.legend(['Эмпирическая', 'Теоретическая'])
    plt.suptitle(f'Функции распределения n={n}')

    plt.savefig(f'{full_path_graphics}/distribution-graphic-{n}.png')

    plt.show()


if __name__ == '__main__':
    get_comparison_table()
    plot_correlograms()
    plot_distribution_density(10000)
    plot_distribution(1000)
