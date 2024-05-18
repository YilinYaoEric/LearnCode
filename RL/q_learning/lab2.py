# try to write everything myself in a more complex version

import numpy as np
import pandas as pd
import time

EPSILON = 0.9
X_LENGTH = 5
Y_LENGTH = 5
ACTIONS = ['left', 'right', 'up', 'down']
REFRESH_TIME = 0.01
EPISODES = 100
GAMMA = 0.95
ALPHA = 0.3

def make_q_table(n_states_x: int, n_states_y, actions: list[int]) -> pd.DataFrame:
    ret = pd.DataFrame(
        np.zeros((n_states_x * n_states_y, len(actions))),
        columns=actions
    )
    return ret

"""
q_table
   left  right   up  down
0   0.0    0.0  0.0   0.0
1   0.0    0.0  0.0   0.0
2   0.0    0.0  0.0   0.0
3   0.0    0.0  0.0   0.0
4   0.0    0.0  0.0   0.0
5   0.0    0.0  0.0   0.0
6   0.0    0.0  0.0   0.0
7   0.0    0.0  0.0   0.0
8   0.0    0.0  0.0   0.0
"""

def make_decision(state: int, q_table: pd.DataFrame) -> str:
    """
    taking the state and q_table, return the choice
    """
    actions: pd.Series = q_table.iloc[state, :]
    if np.random.random() > EPSILON or actions.max() == 0.0:
        choice_name = np.random.choice(ACTIONS)
    else:
        choice_name = ACTIONS[actions.argmax()]
    return choice_name

def env_feedback(action, state):
    """
    taking the choice and state, return the reward and new state
    """
    S_, R = state, -1

    if action == 'left':
        if state % X_LENGTH != 0:
            S_ = state - 1
    elif action == 'right':
        if state % X_LENGTH != X_LENGTH - 1:
            S_ = state + 1
    elif action == 'up':
        if state >= Y_LENGTH:
            S_ = state - Y_LENGTH
    elif action == 'down':
        if state < Y_LENGTH * (X_LENGTH - 1):
            S_ = state + Y_LENGTH
    if S_ == Y_LENGTH * X_LENGTH - 1:
        S_, R = 'terminal', 10
    
    return S_, R

def print_env(state: int, episode: int, step_counter: int) -> None:
    """
    print env interactive
    return True if terminated
    """

    if state == 'terminal':
        interaction = 'Episode: %s, Step_Counter: %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
        return True

    env_str = []
    for _ in range(Y_LENGTH):
        env_str.append(['-'] * X_LENGTH + ['\n'])
    env_str[state//Y_LENGTH][state%Y_LENGTH] = 'o'
    for i in range(Y_LENGTH):
        env_str[i] = ''.join(env_str[i])
    interaction = ''.join(env_str)
    print('\r{}'.format(interaction), end='')
    time.sleep(REFRESH_TIME)
    return False

def rl():
    q_table = make_q_table(X_LENGTH, Y_LENGTH, ACTIONS)

    for episode in range(EPISODES):
        is_ended = False
        steps = 0
        S = 0
        print_env(S, episode, steps)
        steps += 1
        while not is_ended:
            action = make_decision(S, q_table)
            S_, R = env_feedback(action, S)
            q_predict = q_table.loc[S, action]
            is_ended = S_ == 'terminal'
            if S_ == 'terminal':
                q_target = R
            else:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            
            q_table.loc[S, action] += ALPHA * (q_target - q_predict)
            S = S_
            print_env(S, episode, steps)
            steps += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQtable: \n')
    print(q_table)