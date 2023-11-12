import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# WEEK 2
def fun_ex1(t, y):
    return 1 / t ** 2 - y / 2 - y ** 2


def euler_method(interval_start, interval_end, initial_condition, function_name, number_of_nodes=101):
    """
    --Understanding ODEs:--
    Before delving into the Euler method, it's important to understand what an ordinary differential equation (ODE) is.
    An ODE is a mathematical equation that relates the derivative of a function to the function itself. In other words,
    it describes how a function changes concerning its own rate of change.
    A simple example of a first-order ODE is:
        dy/dt = f(t, y)
    where y is the unknown function, t is the independent variable (often representing time), and f(t, y) is some known
    function.

    --Discretizing the Interval:--
    To solve an ODE numerically using the Euler method, you need to discretize the interval over which you want to find
    the solution. This means dividing the interval into smaller time steps. Let's denote the step size as h,
    so the points where you'll approximate the solution will be t_0, t_1, t_2, ..., where t_i = t_0 + ih.

    --Euler's Formula:--
    The Euler method is based on a simple idea: the derivative dy/dt can be approximated as the rate of change over a
    small time interval h. Therefore, you can approximate the change in y over this interval as:
        Δy = hf(t, y)
    This means that the new value of y at the next time step (t_1) can be estimated as:
        y_1 = y_0 + hf(t_0, y_0)

    --Iterative Process:--
    To obtain the solution at subsequent time steps, you repeat the process. For each step i, you calculate y_{i+1}
    using the formula:
        y_{i+1} = y_i + hf(t_i, y_i)

    Advancing in Time: You continue this process until you reach the desired final time, t_final, by incrementing i from
    0 to the number of steps required. The smaller the step size h, the more accurate the approximation will be.

    Limitations: While the Euler method is easy to implement, it has limitations. It is a first-order method,
    which means that the error in the approximation is proportional to the step size h. Therefore, it's not very
    accurate for highly nonlinear or stiff ODEs. Other methods, like the Runge-Kutta methods, can provide better
    accuracy but are more complex.
    :param interval_start:
    :param interval_end:
    :param initial_condition:
    :param function_name: string - the name of the function that has been defined
    :param number_of_nodes:
    :return:
    """
    step_size = (interval_end - interval_start) / (number_of_nodes - 1)
    y = np.zeros(number_of_nodes)
    t = np.linspace(interval_start, interval_end, number_of_nodes)
    y[0] = initial_condition
    for i in range(number_of_nodes - 1):
        try:
            y[i + 1] = y[i] + step_size * eval(function_name)(t[i], y[i])
        except NameError:
            print(f"Function '{function_name}' is not defined.")
            return None
    # Using scipy's odeint to solve the same differential equation for comparison
    t_odeint = np.linspace(interval_start, interval_end, number_of_nodes)
    y_odeint = odeint(eval(function_name), initial_condition, t_odeint)
    # Exact solution
    y_exact = initial_condition / t
    # Calculate relative error
    err_rel = abs(y[-1] - y_exact[-1]) / abs(y_exact[-1])
    # Plot the results
    plt.plot(t, y, 'b', label='Euler Method')
    plt.plot(t, y_exact, 'k', label='Exact Solution')
    plt.plot(t_odeint, y_odeint, '.r', label='ODEINT')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Euler Method vs. Exact Solution vs. ODEINT')
    plt.show()

    print(f'y(N): {y[-1]}')
    print(f'y_exact(N): {y_exact[-1]}')
    print(f'Relative Error: {err_rel}')


euler_method(1, 2, -1, "fun_ex1", 101)
