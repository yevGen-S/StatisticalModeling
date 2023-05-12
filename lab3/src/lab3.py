import matplotlib.pyplot as plt
import os
from LhW import *

sample_size = 10000

absolute_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])

relative_path_graph = "results/graphics"
relative_path_res = "results"

full_path_graphics = os.path.join(absolute_path, relative_path_graph)
full_path_res = os.path.join(absolute_path, relative_path_res)


def to_fixed(number, digits=0):
    return f"{round(number, digits):.{digits}f}"


def calculate_math_expect_and_dispersion(numbers):
    m = sum(numbers) / len(numbers)
    d = sum([x ** 2 for x in numbers]) / len(numbers) - m ** 2
    return [m, d]


def print_math_expect_and_dispersion(task, distr_name, m_th, d_th, m_emp, d_emp):
    print(f' Задание {task} - {distr_name}')
    print(f'| Теоретические: | \t@M@ = {m_th}  | \t@D@ = {d_th}  |')
    print(f'| Эмпирические:  | \t@M@ = {m_emp} | \t@D@ = {d_emp} |')
    print(f'| Погрешность    | \t@M@ = {math.fabs(m_th - m_emp)} | \t@D@ = {math.fabs(d_th - d_emp)} |')
    print('-------------------------------------------------------')


def plot_graphics(task_number, task_name, numbers):
    x_label = 'x'
    y_pdf_label = 'f(x)'
    y_cdf_label = 'F(x)'
    pdf_title = f'Функция плотности распределения - {task_name}'
    cdf_title = f'Функция распределения - {task_name}'

    plt.figure()
    plt.hist(numbers, density=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_pdf_label)
    plt.title(pdf_title)
    plt.savefig(f'{full_path_graphics}/{task_name}-{task_number}-pdf.png')

    plt.figure()
    plt.hist(numbers, density=True, cumulative=True, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_cdf_label)
    plt.title(cdf_title)
    plt.savefig(f'{full_path_graphics}/{task_name}-{task_number}-pdf.png')


def task_1():
    numbers = []
    a = 0
    b = 10

    for _ in range(sample_size):
        numbers.append(rnuni(a, b))

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(1, rnuni.__name__, (a + b) / 2, ((b - a) ** 2) / 12, m_emp, d_emp)

    plot_graphics(1, "uniform_distribution", numbers)


def task_2(distribution):
    numbers = []
    m = 0
    s = 1

    for _ in range(sample_size):
        numbers.append(distribution(m, s))

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(2, distribution.__name__, m, s ** 2, m_emp, d_emp)

    plot_graphics(2, distribution.__name__, numbers)


def task_3():
    numbers = []
    beta = 1

    for _ in range(sample_size):
        numbers.append(rnexp(beta))

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(3, rnexp.__name__, beta, beta ** 2, m_emp, d_emp)

    plot_graphics(3, "exponential_distribution", numbers)


def task_4():
    numbers = []
    n = 10

    for _ in range(sample_size):
        numbers.append(rnchis(n))

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(4, rnchis.__name__, n, 2 * n, m_emp, d_emp)

    plot_graphics(4, "hi2_distribution", numbers)


def task_5():
    numbers = []
    n = 10

    for _ in range(sample_size):
        numbers.append(rnstud(n))

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(5, rnstud.__name__, 0, n / (n - 2), m_emp, d_emp)

    plot_graphics(5, "student_distribution", numbers)


def additional_task_5():
    def erlang_cdf(x):
        return math.exp(-x) * (-(x ** 3) - 3 * (x ** 2) - 6 * x - 6) / 6 + 1

    amount = 100
    beta = 1
    k = 4

    numbers = [rnerlang(beta, k) for _ in range(amount)]

    numbers = sorted(numbers)

    max_variance = max([math.fabs((r + 1) / amount - erlang_cdf(numbers[r])) for r in range(len(numbers))])

    m_emp, d_emp = calculate_math_expect_and_dispersion(numbers)
    print_math_expect_and_dispersion(6, rnerlang.__name__, k / beta, k / beta ** 2, m_emp, d_emp)
    print(f'Максимальный расход:\t{to_fixed(max_variance, 2)}')
    print(f'Критический расход: \t{0.136}')

    plot_graphics(6, "dop_erlang_distribution", numbers)


if __name__ == '__main__':
    task_1()

    task_2(rnnrm1)
    task_2(rnnrm2)

    task_3()

    task_4()

    task_5()

    additional_task_5()
    plt.show()
