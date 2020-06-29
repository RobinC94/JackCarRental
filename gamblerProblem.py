#!/usr/bin/python
# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt


ph = 0.4
discount = 0.9


Tp = np.zeros((101, 101))
R = np.zeros((101,))


def init_trans_prob():
    for i in range(101):
        for j in range(101):
            if i > j:
                Tp[i, j] = 1 - ph
            if i < j and i*2 >= j:
                Tp[i, j] = ph
    R[100] = 1


V = np.zeros((101,))
Action = np.zeros((101,))


def policy_action_evaluate():
    delta = 0
    for i in range(100):
        v = V[i]
        V[i], Action[i] = value_calculate(i)
        delta = max(delta, np.abs(v - V[i]))
    return delta


def value_calculate(i):
    if i == 100:
        return 0, 0
    best_val = 0
    best_act = 0
    i = int(i)
    for a in range(min(i, 100-i)+1):
        temp_v = ph * (R[i+a] + discount * V[i+a]) + (1 - ph) * (R[i-a] + discount * V[i-a])
        if temp_v > best_val:
            best_val = temp_v
            best_act = a
    return best_val, best_act


def plot_value():
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    ax.plot(V)


def plot_action():
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    X = np.arange(101)
    ax.bar(X, Action)


if __name__ == '__main__':
    init_trans_prob()

    print("Evaluate Policies...")
    while 1:
        delta = policy_action_evaluate()
        print(Action)
        print(V)
        if delta == 0:
            print("Evaluate Finished!")
            break
    plot_value()
    plot_action()
    plt.show()

