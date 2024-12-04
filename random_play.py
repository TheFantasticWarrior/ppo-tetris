import tetris
import numpy as np
x = tetris.Container()
iterations = 100000
acs = np.random.randint(10, size=(iterations, 2))
rews = np.zeros((iterations, 2))
for i in range(iterations):
    x.step(*acs[i])
    _, done, rew = x.get_state()
    rews[i] = rew
    if done:
        x.reset()
        breakpoint()
        rews[i]=0
print(rews.mean())
breakpoint()
