import gym
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def q_learn(l, r):
    env = gym.make('Taxi-v3')
    episodes = 1000
    if r:
       episodes = 1
    dataTable = np.zeros((episodes, 2))
    if l:
        qTable = np.loadtxt('qtable.csv', delimiter=',')
    else:
        qTable = np.zeros((500, 6))
    alpha = .1
    gamma = .8
    for i in range(episodes):
        dataTable[i][0], dataTable[i][1] = run_episode(env, qTable, alpha, gamma, r)
    np.savetxt('qtable.csv', qTable, delimiter=',')
    np.savetxt('qdata.csv', dataTable, delimiter=',')


def run_episode(env, qTable, alpha, gamma, r):
    state = env.reset()
    if r:
        env.render()
    env._max_episode_steps = 3000
    done = False
    i = 0
    while not done:
        s = state
        action = qTable[state].argmax()
        state, reward, done, P = env.step(action)
        qTable = update_table(s, action, reward, gamma, alpha, qTable, env)
        if r:
            env.render()
        i += 1
    print("last reward: " + str(reward) + " | " + "steps: " + str(i))
    return reward, i


def get_successors(s_idx, action, env):
    taxi_row, taxi_col, passenger, dest = env.decode(s_idx)
    state = (taxi_row, taxi_col, passenger, dest) 
    locs = [0, 4, 20, 24]
    next_state = state
    idx = 5 * taxi_row + taxi_col
    if action == 0:
        next_state = check_south(idx, state)
    elif action == 1:
        next_state = check_north(idx, state)
    elif action == 2:
        next_state = check_east(idx, state)
    elif action == 3:
        next_state = check_west(idx, state)
    elif action == 4: 
        if (passenger < 4 and idx == locs[passenger]):
            next_state = (taxi_row, taxi_col, 4, dest)
    else:
        if (passenger == 4 and idx in locs):
            new_pas = locs.index(idx)
            next_state = (taxi_row, taxi_col, new_pas, dest)
    return next_state


def check_west(idx, state):
    valid_idx = [1, 3, 4, 6, 8, 9, 11, 12, 13, 14, 17, 19, 22, 24]
    if idx in valid_idx:
        return (state[0], state[1] - 1, state[2], state[3])
    return state

    
def check_east(idx, state):
    valid_idx = [0, 2, 3, 5, 7, 8, 10, 11, 12, 13, 16, 18, 21, 23]
    if idx in valid_idx:
        return (state[0], state[1] + 1, state[2], state[3])
    return state


def check_south(idx, state):
    valid_idx = [i for i in range(20)]
    if idx in valid_idx:
        return (state[0] + 1, state[1], state[2], state[3])
    return state


def check_north(idx, state):
    valid_idx = [i for i in range(5, 25)]
    if idx in valid_idx:
        return (state[0] - 1, state[1], state[2], state[3])
    return state


def update_table(state, action, reward, gamma, alpha, qTable, env):
    old = qTable[state][action]
    next_state = get_successors(state, action, env)
    s_idx = env.encode(next_state[0], next_state[1], next_state[2], next_state[3]) 
    best = qTable[s_idx].max()
    new = alpha* (reward + gamma*best)
    qTable[state][action] = (1 - alpha) * old + new
    return qTable


if __name__=="__main__":
    num_args = len(sys.argv)
    l, r = False, False
    if "-l" in sys.argv:
        l = True
    if "-r" in sys.argv:
        r = True
    q_learn(l, r)


