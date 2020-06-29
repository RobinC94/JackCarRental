import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()


rent_income = 10
move_cost = 2
discount = 0.9
MAX_CAR_GARAGE = 20
MAX_CAR_MOVE = 5
lambda_rent = [3, 4]
lambda_return = [3, 2]
accurate = 1e-6

Tp = np.zeros((2, 21, 21))
R = np.zeros((2, 21))


def possion_prob(lam, n):
    return np.exp(-lam) * (lam ** n) / np.math.factorial(n)


def trans_prob(s, garage):
    for r in range(0, MAX_CAR_GARAGE + 1):
        p_rent = possion_prob(lambda_rent[garage], r)
        if p_rent < accurate:
            return
        rent = min(s, r)
        R[garage, s] += p_rent * rent_income * rent
        for ret in range(0, MAX_CAR_GARAGE + 1):
            p_ret = possion_prob(lambda_return[garage], ret)
            if p_ret < accurate:
                continue
            s_next = min(s - rent + ret, MAX_CAR_GARAGE)
            Tp[garage, s, s_next] += p_rent * p_ret


def init_trans_prob():
    for i in range(0, MAX_CAR_GARAGE + 1):
        trans_prob(i, 0)
        trans_prob(i, 1)


V = np.zeros((21, 21))
Action = np.zeros((21, 21))


def policy_evaluate():
    delta = 0
    for i in range(0, MAX_CAR_GARAGE + 1):
        for j in range(0, MAX_CAR_GARAGE + 1):
            v = V[i, j]
            temp_v = 0
            for m in range(0, MAX_CAR_GARAGE + 1):
                for n in range(0, MAX_CAR_GARAGE + 1):
                    temp_v += Tp[0, i, m] * Tp[1, j, n] * (R[0, i] + R[1, j] + discount * V[m, n])
            V[i, j] = temp_v
            delta = max(delta, np.abs(v - V[i, j]))
    return delta


def policy_evaluate2():
    delta = 0
    for i in range(0, MAX_CAR_GARAGE + 1):
        for j in range(0, MAX_CAR_GARAGE + 1):
            v = V[i, j]
            a = Action[i, j]
            V[i, j] = value_calculate(i, j, a)
            delta = max(delta, np.abs(v - V[i, j]))
    return delta


def value_calculate(i, j, a):
    if a > i:
        a = i
    elif a < 0 and -a > j:
        a = -j
    ii = int(i - a)
    jj = int(j + a)
    ii = min(ii, MAX_CAR_GARAGE)
    jj = min(jj, MAX_CAR_GARAGE)
    temp_v = -np.abs(a) * move_cost
    for m in range(0, MAX_CAR_GARAGE+1):
        for n in range(0, MAX_CAR_GARAGE+1):
            temp_v += Tp[0, ii, m] * Tp[1, jj, n] * (R[0, ii] + R[1, jj] + discount*V[m, n])
    return temp_v


def action_greedy(i, j):
    best_action = 0
    best_value = 0
    for a in range(-MAX_CAR_MOVE, MAX_CAR_MOVE+1):
        if a > i:
            continue
        elif a < 0 and -a > j:
            continue
        val = value_calculate(i, j, a)
        if val > best_value + 0.1:
            best_value = val
            best_action = a
    return best_action


def policy_improve():
    stable_flag = True
    for i in range(0, MAX_CAR_GARAGE+1):
        for j in range(0, MAX_CAR_GARAGE+1):
            act_best = action_greedy(i, j)
            if act_best != Action[i, j]:
                Action[i, j] = act_best
                stable_flag = False
    return stable_flag


def plot_value1():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Income')
    aZ = []
    aX = []
    aY = []
    for i in range(MAX_CAR_GARAGE + 1):
        for j in range(MAX_CAR_GARAGE + 1):
            aX.append(i)
            aY.append(j)
            aZ.append(V[i, j])
    ax.set_ylabel('# of cars at location 1')
    ax.set_xlabel('# of cars at location 2')
    ax.scatter(aX, aY, aZ)


def plot_value2():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(0, MAX_CAR_GARAGE + 1), range(0, MAX_CAR_GARAGE + 1))
    ax.scatter(X, Y, V)


def print_policy(p, i=''):
    plt.figure()
    ticks = [0] + [''] * (MAX_CAR_GARAGE - 1) + [MAX_CAR_GARAGE]
    ax = sns.heatmap(p.astype(int), square=True, xticklabels=ticks, yticklabels=ticks)
    ax.set_title('Policy ' + str(i))
    ax.set_ylabel('# of cars at location 1')
    ax.set_xlabel('# of cars at location 2')
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(MAX_CAR_MOVE * 2 + 1) - MAX_CAR_MOVE)
    cbar.set_ticklabels(np.arange(MAX_CAR_MOVE * 2 + 1) - MAX_CAR_MOVE)


if __name__ == '__main__':
    init_trans_prob()
    stable = False
    policies = []
    count = 0
    while not stable:
        print("Evaluate Policies...")
        while 1:
            delta = policy_evaluate2()
            if delta < 0.1:
                print("Evaluate Finished!")
                break
        print("Improve Policies...")
        count += 1
        stable = policy_improve()
        policies.append(Action.copy())
        print_policy(Action, str(count))
    plot_value2()
    plt.show()





