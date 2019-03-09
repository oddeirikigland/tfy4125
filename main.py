import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m = 0.1

# ip_track - interpolate track
#
# SYNTAX
# p=ip_track(filename)
#
# INPUT
# filename: data file containing exported tracking data on the standard
# Tracker export format
#
# mass_A
# t	x	y
# 0.0	-1.0686477620876644	42.80071293284619
# 0.04	-0.714777136706708	42.62727536827738
# ...
#
# OUTPUT
# p=ip_track(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.
def ip_track(filename):
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


# tr_values - track values
#
# SYNTAX
# [y,dydx,d2ydx2,alpha,R]=tr_values(p,x)
#
# INPUT
# p: the n+1 coefficients of a polynomial of degree n, given in descending
# order. (For instance the output from p=ipt_rack(filename).)
# x: ordinate value at which the polynomial is evaluated.
#
# OUTPUT
# [y,dydx,d2ydx2,alpha,R]=tr_values(p,x) returns the value y of the
# polynomial at x, the derivative dydx and the second derivative d2ydx2 in
# that point, as well as the slope alpha(x) and the radius of the
# osculating circle.
# The slope angle alpha is positive for a curve with a negative derivative.
# The sign of the radius of the osculating circle is the same as that of
# the second derivative.
def tr_values(p, x):
    y = np.polyval(p, x)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x)
    alpha = np.arctan(-dydx)
    R = (1.0 + dydx ** 2) ** 1.5 / d2ydx2
    return y, dydx, d2ydx2, alpha, R


def plot_raw_data(filename):
    data = np.loadtxt(filename, skiprows=2)

    # plots s(t)
    plt.figure()
    plt.xlabel("Posisjon x [m]")
    plt.ylabel("Posisjon y [m]")
    plt.plot(data[:, 1], data[:, 2])

    deltas = np.diff(data, axis=0)

    # plots v(t)
    plt.figure()
    plt.xlabel("Tid t [s]")
    plt.ylabel("Fart v [m/s]")
    plt.plot(
        data[:-1, 0],
        np.sqrt(
            (deltas[:, 1] / deltas[:, 0]) ** 2 + (deltas[:, 2] / deltas[:, 0]) ** 2
        ),
    )


def slope_acceleration(angle_of_inclination):
    aks = 5 * g * np.sin(angle_of_inclination) / 7
    aks_x = aks * np.cos(angle_of_inclination)
    aks_y = aks * np.sin(angle_of_inclination)
    return np.sqrt(aks_x ** 2 + aks_y ** 2)


def calculate_numeric_values(filename):
    poly = ip_track(filename)

    t = np.array([])
    x = np.array([])
    y = np.array([])
    v = np.array([])
    normal_force = np.array([])
    friction_force = np.array([])

    # step size
    h = 0.001

    # initial values
    t_old = 0
    x_old = 0.7393787749776366
    v_old = 0

    while x_old > -0.3305748273167307:
        y_old, dydx, d2ydx2, alpha, R = tr_values(poly, x_old)

        t = np.append(t, t_old)
        x = np.append(x, x_old)
        y = np.append(y, y_old)
        v = np.append(v, v_old)

        x_new = x_old - h * v_old * np.cos(alpha)

        if x_old > 0:
            # ball in slope
            acceleration = slope_acceleration(angle_of_inclination=alpha)
            v_new = v_old + h * acceleration
            normal_force_value = m * (v_old ** 2 / R + g * np.cos(alpha))
            friction_force_value = m * (g * np.sin(alpha) - acceleration)
        else:
            # ball have jumped
            v_new = v_old + h * g
            normal_force_value = 0
            friction_force_value = 0

        t_old += h
        x_old = x_new
        v_old = v_new
        normal_force = np.append(normal_force, normal_force_value)
        friction_force = np.append(friction_force, friction_force_value)
    return t, x, y, v, normal_force, friction_force


def plot_numeric_result(t, x, y, v, normal_force, friction_force):
    # plots s(t)
    plt.figure()
    plt.xlabel("Posisjon x [m]")
    plt.ylabel("Posisjon y [m]")
    plt.plot(x, y)

    # plots v(t)
    plt.figure()
    plt.xlabel("Tid t [s]")
    plt.ylabel("Fart v [m/s]")
    plt.plot(t, v)

    # plots normal force
    plt.figure()
    plt.xlabel("Tid t [s]")
    plt.ylabel("Normalkraft N [N]")
    plt.plot(t, normal_force)

    # plots friction force
    plt.figure()
    plt.xlabel("Tid t [s]")
    plt.ylabel("Friction f [N]")
    plt.plot(t, friction_force)


def main():
    filename = "resultater_video8.txt"
    plot_raw_data(filename)
    plot_numeric_result(*calculate_numeric_values(filename))
    plt.show()


if __name__ == "__main__":
    main()
