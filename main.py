import numpy as np
import matplotlib.pyplot as plt

g = 9.81

# iptrack - interpolate track
#
# SYNTAX
# p=iptrack(filename)
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
# p=iptrack(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.
def iptrack(filename):
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:, 1], data[:, 2], 15)


# trvalues - track values
#
# SYNTAX
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x)
#
# INPUT
# p: the n+1 coefficients of a polynomial of degree n, given in descending
# order. (For instance the output from p=iptrack(filename).)
# x: ordinate value at which the polynomial is evaluated.
#
# OUTPUT
# [y,dydx,d2ydx2,alpha,R]=trvalues(p,x) returns the value y of the
# polynomial at x, the derivative dydx and the second derivative d2ydx2 in
# that point, as well as the slope alpha(x) and the radius of the
# osculating circle.
# The slope angle alpha is positive for a curve with a negative derivative.
# The sign of the radius of the osculating circle is the same as that of
# the second derivative.
def trvalues(p, x):
    y = np.polyval(p, x)
    dp = np.polyder(p)
    dydx = np.polyval(dp, x)
    ddp = np.polyder(dp)
    d2ydx2 = np.polyval(ddp, x)
    alpha = np.arctan(-dydx)
    R = (1.0 + dydx ** 2) ** 1.5 / d2ydx2
    return y, dydx, d2ydx2, alpha, R


def plotRawData(filename):
    data = np.loadtxt(filename, skiprows=2)

    # plots s(t)
    plt.figure()
    plt.xlabel("Posisjon x [m]")
    plt.ylabel("Posisjon y [m]")
    plt.plot(data[:, 1], data[:, 2])

    # plots v(t)
    deltas = np.diff(data, axis=0)

    plt.figure()
    plt.xlabel("Tid t [s]")
    plt.ylabel("Fart v [m/s]")
    plt.plot(
        data[:-1, 0],
        np.sqrt(
            (deltas[:, 1] / deltas[:, 0]) ** 2 + (deltas[:, 2] / deltas[:, 0]) ** 2
        ),
    )


def slope_akseleration(angle_of_inclination):
    return 5 * g * np.sin(angle_of_inclination) / 7


# regn ut numerisk: s(t), v(t), normalkraft, friksjonskraft
def calculate(filename):
    poly = iptrack(filename)

    t = np.array([])
    x = np.array([])
    y = np.array([])
    v = np.array([])

    N = 100000  # number of steps
    h = 0.001  # step size
    i = 0

    # initial values
    t_old = 0
    x_old = 0.7393787749776366
    v_old = 0

    # t = np.zeros(N+1)
    # x = np.zeros(N+1)
    # y = np.zeros(N+1)
    # v = np.zeros(N+1)
    # t[0] = t_0
    # v[0] = v_0

    while x_old > -0.3305748273167307:
        y_old, dydx, d2ydx2, alpha, R = trvalues(poly, x_old)

        print(alpha)

        t = np.append(t, t_old)
        x = np.append(x, x_old)
        y = np.append(y, y_old)
        v = np.append(v, v_old)

        x_new = x_old + h * v_old

        if x_old > 0:
            # ball in slope
            v_new = v_old + h * slope_akseleration(angle_of_inclination=alpha)
        else:
            # ball have jumped
            v_new = v_old - h * g

        t_old = t_old + h
        x_old = x_new
        v_old = v_new

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
    print(v)


def main():
    filename = "resultater_video8.txt"
    plotRawData(filename)
    calculate(filename)
    plt.show()


if __name__ == "__main__":
    main()
