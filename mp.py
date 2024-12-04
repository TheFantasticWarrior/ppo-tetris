
import numpy as np
import torch.multiprocessing as mp
import tetris
from torch.multiprocessing import Pipe, Process


class manager:
    def __init__(self, nenvs, seed, render) -> None:
        cpus = mp.cpu_count()
        self.nenvs = nenvs-(nenvs % cpus) if nenvs > cpus else nenvs
        self.nremotes = min(cpus, self.nenvs)
        self.seeds = np.arange(self.nenvs)+seed
        # self.envs=[make_env(seed+i) for i in range(nenvs)]
        env_seeds = np.array_split(self.seeds, self.nremotes)
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=gather_data,
                           args=(work_remote, remote,
                                 render and (seeds[0] == seed), seeds))
                   for work_remote, remote, seeds in
                   zip(self.work_remotes, self.remotes, env_seeds)]
        for p in self.ps:
            p.daemon = True
            p.start()
        self.count = 0
        self.lines = 0
        self.atk = 0
        for remote in self.work_remotes:
            remote.close()

    def send(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def recv(self):
        ob, rew, done, info = zip(* [remote.recv()
                                     for remote in self.remotes])
        obs = _flatten_obs(ob)
        c = np.stack(rew).T
        for i in info:
            if i is not None:
                self.count += 2
                self.lines += i[0]+i[2]
                self.atk += i[1]+i[3]
        return obs, done, c

    def reset_data(self):
        self.count = 0
        self.lines = 0
        self.atk = 0

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


def gather_data(remote, parent_remote, render, seed):
    parent_remote.close()
    game = tetris.Container()
    game.seed_reset(seed[0])
    old_filled = np.array(game.check_filled())
    if render:
        r = tetris.Renderer(1, 10)
    try:
        while True:

            cmd, action = remote.recv()
            if cmd == 'step':
                game.step(*action)
                x, done, rew = game.get_state()
                # filled = np.array(game.check_filled())
                # filled_diff = filled-old_filled
                # old_filled = filled
                # rew = [rew[i]+(((filled_diff[i]) +
                #        np.sign(filled[i] - 0.5))*0.0001
                #        if action[i] == 1 else 0)
                #        for i in range(2)]
                if render:
                    r.render(game)
                # if rew[0]<-0.002:
                #     print(f"{rew[0]:.4f}")
                if done:
                    info = game.reset()
                    x, _, _ = game.get_state()
                    # old_filled = np.array(game.check_filled())
                else:
                    info = None
                remote.send((x, rew, done, info))
            elif cmd == 'reset':
                x, _, _ = game.get_state()
                remote.send(x)
            elif cmd == 'close':
                remote.close()
                if render:
                    r.close()
                break
    except KeyboardInterrupt:
        pass


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    # (nenv,2,5,...)->(5,2,nenv,...)
    return [np.swapaxes(np.stack(
        [[player[i] for player in env] for env in obs]), 0, 1)
        for i in range(5)]
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    try:
        assert isinstance(l, (list, tuple))
        assert len(l) > 0
        assert all([len(l_) > 0 for l_ in l])
    except Exception as e:
        [print(len(l_), l_) for l_ in l]
        raise e
    return [l__ for l_ in l for l__ in l_]
