import torch


class Buffer:
    def __init__(self, nsteps, nenvs):

        self.obs = [torch.zeros((nsteps, nenvs, 2, 3)),
                    torch.zeros((nsteps, nenvs, 2, 7)),
                    torch.zeros((nsteps, nenvs, 2, 7)),
                    torch.zeros((nsteps, nenvs, 2, 22, 12)),
                    torch.zeros((nsteps, nenvs, 2, 10))
                    ]
        self.done = torch.zeros((nsteps, nenvs, 2))
        self.rews = torch.zeros((nsteps, nenvs, 2))
        self.actions = torch.zeros((nsteps, nenvs, 2))
        self.logprob = torch.zeros((nsteps, nenvs, 2))
        self.values = torch.zeros((nsteps, nenvs, 2))


def entropy_and_logprob(logits, action):
    probs = torch.distributions.categorical.Categorical(logits=logits)
    return probs.entropy(), probs.logprob(action)


def sample(logits):
    probs = torch.distributions.categorical.Categorical(logits=logits)
    return probs.sample()
