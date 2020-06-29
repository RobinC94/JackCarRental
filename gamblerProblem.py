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


def policy_evaluate():
    delta = 0
    for i in range(101):
        v = V[i]
        a = Action[i]
        V[i] = value_calculate(i)
        delta = max(delta, np.abs(v - V[i]))
    return delta


def value_calculate(i):
    if i == 100:
        return 1
    temp_v = 0
    i = int(i)
    count = 0
    for a in range(min(i, 100-i)+1):
        temp_v += ph * (R[i+a] + discount * V[i+a])
        temp_v += (1 - ph) * (R[i-a] + discount * V[i-a])
        count += 1
    if count > 0:
        temp_v /= count
    return temp_v


def action_greedy(i):
    best_action = 0
    best_value = 0
    for a in range(min(i, 100-i)+1):
        val = ph * V[i+a] + (1-ph) * V[i-a]
        if val > best_value:
            best_value = val
            best_action = a
    return best_action


def policy_improve():
    stable_flag = True
    for i in range(101):
        act_best = action_greedy(i)
        if act_best != Action[i]:
            Action[i] = act_best
            stable_flag = False
    return stable_flag


def plot_value():
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    ax.plot(V)


if __name__ == '__main__':
    init_trans_prob()
    stable = False
    policies = []
    count = 0
    while not stable:
        print("Evaluate Policies...")
        while 1:
            delta = policy_evaluate()
            if delta == 0:
                print("Evaluate Finished!")
                break
        print("Improve Policies...")
        count += 1
        stable = policy_improve()
        policies.append(Action.copy())
        print(Action)
        print(V)
    plot_value()
    plt.show()

