"""
自己尝试只靠公式复刻Sarsa
"""

import pandas as pd
import numpy as np

class QTable:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epislon=0.95):
        self.actions: list = actions
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epislon: float = epislon
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def pick_action(self, state) -> str:
        """
        given state return action
        """
        state = str(state)
        self._ensure_state_exist(state)
        argmax = self.q_table.loc[state, :].argmax()
        if np.random.random() > self.epislon or argmax == 0:
            return np.random.choice(self.actions)
        return self.actions[argmax]

    def learn(self, s, s_, a, a_, r):
        pass

    def _ensure_state_exist(self, s):
        if s not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    np.zeros((len(self.actions))),
                    index=self.q_table.columns,
                    name=s
                )
            )

class Sarsa0(QTable):
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        super(Sarsa0, self).__init__(actions, alpha, gamma)

    def learn(self, s, s_, a, a_, r):
        """
        given s, s_, a, a_, and r
        update the q tabel
        """
        s, s_ = str(s), str(s_)
        self._ensure_state_exist(s_)

        predict = r + self.gamma * self.q_table.loc[s_, a_] if s != 'terminal' else r 
        self.q_table.loc[s, a] += (
            self.alpha * (predict - self.q_table.loc[s, a])
        )
        print(self.q_table)

# try myself based on the equation
class SarsaLambda(QTable):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epislon=0.95, lambda_ = 0.9):
        super(SarsaLambda, self).__init__(actions, alpha, gamma, epislon)
        self.lambda_ = lambda_
        self.e_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def learn(self, s, s_, a, a_, r):
        """
        given s, s_, a, a_, and r
        update the q table and the e table
        """
        s, s_ = str(s), str(s_)
        self._ensure_state_exist(s_)
        predict = r + self.gamma * self.q_table.loc[s_, a_] if s != 'terminal' else r 
        update = self.alpha * (predict - self.q_table.loc[s, a])

        self.e_table.loc[s, a] += 1
        for temp_s in self.e_table.index:
            for temp_a in self.e_table.columns:
                self.q_table.loc[temp_s, temp_a] += update * self.e_table.loc[temp_s, temp_a]
                e = self.e_table.loc[temp_s, temp_a] 
                self.e_table.loc[temp_s, temp_a] = min(
                    self.gamma * self.lambda_ * e, 3
                    )
        

    def _ensure_state_exist(self, s):
        if s not in self.e_table.index:
            self.e_table = self.e_table.append(
                pd.Series(
                    np.zeros((len(self.actions))),
                    index = self.e_table.columns,
                    name = s
                )
            )
        super()._ensure_state_exist(s)