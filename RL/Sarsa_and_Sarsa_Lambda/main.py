from maze_env import Maze
from qtable import Sarsa0
from qtable import SarsaLambda

def update():
    for episode in range(100):
        s = env.reset()
        a = RL.pick_action(s)
        
        while True:
            env.render()
            # a = RL.pick_action(s, epislon = min(0.9, (1-1/(episode + 1))))
            s_, r, done = env.step(a)
            # a_ = RL.pick_action(s_, epislon = min(0.9, (1-1/(episode + 1))))
            a_ = RL.pick_action(s_)
            RL.learn(s, s_, a, a_, r)
            print(a)
            s = s_
            a = a_

            if done:
                break

    print('Game Ended')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    # RL = Sarsa0(actions=list(range(env.n_actions)))
    RL = SarsaLambda(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()