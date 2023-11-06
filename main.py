import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# WEEK 2
def funEx1(t, y):
    return 1 / t ** 2 - y / 2 - y ** 2


def EulerMethod(interval_start, interval_end, initial_condition, function_name, number_of_nodes=101):
    stepSize = (interval_end - interval_start) / (number_of_nodes - 1)
    y = np.zeros(number_of_nodes)
    t = np.linspace(interval_start, interval_end, number_of_nodes)
    y[0] = initial_condition
    for i in range(number_of_nodes):
        try:
            y[i + 1] = y[i] + stepSize * eval(function_name)(t[i], y[i])
        except NameError:
            print(f"Function '{function_name}' is not defined.")
            return None
    # Using scipy's odeint to solve the same differential equation for comparison
    t_odeint = np.linspace(interval_start, interval_end, number_of_nodes)
    y_odeint = odeint(eval(function_name), initial_condition, t_odeint)