import csv
import os
from LhW import *
import matplotlib.pyplot as plt
import numpy as np
import mpmath

absolute_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])

relative_path_graph = "results/graphics"
relative_path_res = "results"

full_path_graphics = os.path.join(absolute_path, relative_path_graph)
full_path_res = os.path.join(absolute_path, relative_path_res)


def calculate_math_expect_and_dispersion(numbers):
    m = sum(numbers) / len(numbers)
    d = sum([x ** 2 for x in numbers]) / len(numbers) - m ** 2
    return [m, d]


def show_plot(numbers, task=1, **kwargs):
    amount = len(numbers)
    r_low = kwargs.get('r_low', 1)
    r_up = kwargs.get('r_up', 100)
    n = kwargs.get('n', 10)
    p = kwargs.get('p', 0.5)
    mu = kwargs.get('mu', 10)
    number_in_task = kwargs.get('number_in_task', 1)

    tasks = ['uniform distribution', 'binomial distribution', 'geom distribution', 'poisson distribution']

    x = []
    y_pdf = []
    y_cdf = []

    x_label = 'x'
    y_pdf_label = 'f(x)'
    y_cdf_label = 'F(x)'

    legend = ['Теоретическая', 'Эмпирическая']
    pdf_title = f'Функция плотности распределения {tasks[task - 1]}-{task}'
    cdf_title = f'Функция распределения {tasks[task - 1]}-{task}'

    if task == 1:
        x = np.linspace(r_low, r_up, amount)
        y_pdf = [1 / (r_up - r_low)] * amount
        y_cdf = [(k - r_low) / (r_up - r_low) for k in x]

    elif task == 2:
        x = [k for k in range(n + 1)]
        y_pdf = [math.comb(n, k) * (p ** k) * (1 - p) ** (n - k) for k in x]
        y_cdf = [sum([math.comb(n, i) * (p ** i) * (1 - p) ** (n - i) for i in range(k + 1)]) for k in x]

    elif task == 3:
        x = [k for k in range(min(numbers), max(numbers) + 1)]
        y_pdf = [((1 - p) ** (k - 1)) * p for k in x]
        y_cdf = [1 - (1 - p) ** k for k in x]

    elif task == 4:
        x = [k for k in range(min(numbers), max(numbers) + 1)]
        y_pdf = [math.exp(-mu) * (mu ** k) / math.factorial(k) for k in x]
        y_cdf = [mpmath.gammainc(k + 1, mu) / math.factorial(k) for k in x]

    plt.figure()
    plt.hist(numbers, density=True, edgecolor='black')
    plt.plot(x, y_pdf, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.legend(legend)
    plt.title(pdf_title)
    plt.savefig(f'{full_path_graphics}/{tasks[task - 1]}-{task}-{number_in_task}-pdf.png')

    plt.figure()
    plt.hist(numbers, density=True, cumulative=True, edgecolor='black')
    plt.plot(x, y_cdf, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.legend(legend)
    plt.title(cdf_title)

    plt.savefig(f'{full_path_graphics}/{tasks[task - 1]}-{task}-{number_in_task}-cdf.png')


def print_math_expect_and_dispersion(task, distr_name, m_th, d_th, m_emp, d_emp):
    print(f' Задание {task} - {distr_name}')
    print(f'| Теоретические: | \t@M@ = {m_th}  | \t@D@ = {d_th}  |')
    print(f'| Эмпирические:  | \t@M@ = {m_emp} | \t@D@ = {d_emp} |')
    print('-------------------------------------------------------')


def task_1():
    r_low = 1
    r_up = 100

    amount = 10000
    numbers = []

    n = r_up - r_low + 1

    for _ in range(amount):
        numbers.append(irnuni(r_low, r_up))

    show_plot(numbers, task=1, lower=r_low, upper=r_up)
    print_math_expect_and_dispersion(1, irnuni.__name__, (r_low + r_up) / 2, (n ** 2 - 1) / 12,
                                     *calculate_math_expect_and_dispersion(numbers))


def task_2():
    n = 10
    p = 0.5
    amount = 10000
    numbers = []

    for _ in range(amount):
        numbers.append(irnbin(n, p))

    show_plot(numbers, task=2, n=n, p=p)
    print_math_expect_and_dispersion(2, irnbin.__name__, n * p, n * p * (1 - p),
                                     *calculate_math_expect_and_dispersion(numbers))


def task_3(geometrical_distribution, number):
    p = 0.5
    amount = 10000
    numbers = []

    for _ in range(amount):
        numbers.append(geometrical_distribution(p))

    show_plot(numbers, task=3, p=p, number_in_task=number)
    print_math_expect_and_dispersion(3, geometrical_distribution.__name__, 1 / p, (1 - p) / p ** 2,
                                     *calculate_math_expect_and_dispersion(numbers))


def task_4(poisson_distribution, number):
    mu = 10.0
    amount = 10000
    numbers = []

    for _ in range(amount):
        numbers.append(poisson_distribution(mu))

    show_plot(numbers, task=4, mu=mu, number_in_task=number)
    print_math_expect_and_dispersion(4, poisson_distribution.__name__, mu, mu,
                                     *calculate_math_expect_and_dispersion(numbers))


def additional_task_4():
    amount = 100
    mu = 4
    intervals_amount = math.ceil(1 + math.log2(amount))
    distrs = [irnpoi, irnpsn]

    for i in range(2):
        poisson_distribution = distrs[i]
        numbers = []

        for _ in range(amount):
            numbers.append(poisson_distribution(mu))

        show_plot(numbers, task=4, mu=mu)

        interval_width = (max(numbers) - min(numbers)) / (intervals_amount - 1)
        intervals = []
        intervals_centers = []
        frequencies = []
        f_frequencies = []
        frequencies_dispersion = []
        xv_sum = 0
        left = min(numbers) - interval_width / 2

        for k in range(intervals_amount):
            i_left = left + interval_width * k
            i_right = i_left + interval_width
            center = (i_left + i_right) / 2
            frequency = len([x for x in numbers if i_left <= x < i_right])
            xv_sum += center * frequency
            intervals.append(tuple([i_left, i_right]))
            intervals_centers.append(center)
            frequencies.append(frequency)

        m = xv_sum / amount

        for k in range(intervals_amount):
            i_p = (m ** round(intervals_centers[k])) * math.exp(-m) / math.factorial(round(intervals_centers[k]))
            f_frequencies.append(amount * i_p)

        new_intervals_amount = intervals_amount
        k = 0
        while k < new_intervals_amount:
            # склеивание интервалов, если частота меньше 5
            if frequencies[k] < 5:
                if k < new_intervals_amount - 1:
                    frequencies = [*frequencies[0:k], frequencies[k] + frequencies[k + 1], *frequencies[k + 2:]]
                    f_frequencies = [*f_frequencies[0:k], f_frequencies[k] + f_frequencies[k + 1],
                                     *f_frequencies[k + 2:]]
                else:
                    k -= 1
                    frequencies = [*frequencies[0:k], frequencies[k] + frequencies[k + 1]]
                    f_frequencies = [*f_frequencies[0:k], f_frequencies[k] + f_frequencies[k + 1]]

                new_intervals_amount -= 1
            else:
                k += 1

        for k in range(new_intervals_amount):
            frequencies_dispersion.append(((frequencies[k] - f_frequencies[k]) ** 2) / f_frequencies[k])

        print(f'Алгоритм {i + 1}:')
        print(f'\tНаблюдаемое значение хи-квадрат: {sum(frequencies_dispersion)}')
        print(f'\tКритическое значение хи-квадрат: 124.342\n')


if __name__ == '__main__':
    # Равномерное распределение
    task_1()

    # # Биномиальное распределение
    task_2()

    # Геометрическое распределение (3 алгоритма генерации)
    task_3(irngeo_1, 1)
    task_3(irngeo_2, 2)
    task_3(irngeo_3, 3)

    # Пуассоновское распределение (2 алгоритма генерации)
    task_4(irnpoi, 1)
    task_4(irnpsn, 2)

    additional_task_4()
    plt.show()
