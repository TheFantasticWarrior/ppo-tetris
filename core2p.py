import torch
import args

class Buffer:
    def __init__(self, nsteps, nenvs):

        self.obs = [torch.zeros((2, nsteps, nenvs, 4), dtype=torch.float32).to(torch.device("cuda")),
                    torch.zeros((2, nsteps, nenvs, 7), dtype=torch.int).to(
                        torch.device("cuda")),
                    torch.zeros((2, nsteps, nenvs, 7), dtype=torch.int).to(
                        torch.device("cuda")),
                    torch.zeros((2, nsteps, nenvs, 22, 12)).to(torch.device("cuda"),
                                                               dtype=torch.float32),
                    torch.zeros((2, nsteps, nenvs, 10), dtype=torch.float32).to(
                        torch.device("cuda"))
                    ]
        self.memory = torch.zeros(
            (2, nsteps, nenvs, 6, args.d_model), dtype=torch.float32).to(torch.device("cuda"))
        self.done = torch.zeros((nsteps, nenvs)).to(torch.device("cuda"))
        self.rews = torch.zeros((2, nsteps, nenvs)).to(torch.device("cuda"))
        self.actions = torch.zeros((2, nsteps, nenvs)).to(torch.device("cuda"))
        self.logprob = torch.zeros((2, nsteps, nenvs)).to(torch.device("cuda"))
        self.values = torch.zeros((2, nsteps, nenvs)).to(torch.device("cuda"))
        self.size = nsteps*nenvs

    def record_obs(self, nstep, obs):
        for i in range(5):
            self.obs[i][:, nstep] = torch.tensor(obs[i])

    def flatten(self):
        l = [self.actions, self.logprob, self.values, self.memory]
        flat_obs = [[x.view(-1, *x.shape[3:]),
                     x.flip(dims=[0]).view(-1, *x.shape[3:])] for x in self.obs]
        flat_l = [x.view(-1, *x.shape[3:]) for x in l]
        return flat_obs, flat_l


def entropy_and_logprob(logits, action):
    probs = torch.distributions.categorical.Categorical(logits=logits)
    return probs.entropy(), probs.log_prob(action)


def sample(logits):
    probs = torch.distributions.categorical.Categorical(logits=logits)
    action = probs.sample()
    return action, probs.log_prob(action)


def calc_gae(buf, next_vals, last_done, gamma=args.gamma, gae_lambda=0.95):
    advantages = torch.zeros_like(buf.rews).to(torch.device("cuda"))
    lastgaelam = 0
    num_steps = advantages.size(1)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = ~last_done
            nextvalues = next_vals
        else:
            nextnonterminal = 1.0 - buf.done[t + 1]
            nextvalues = buf.values[:, t + 1]
        delta = buf.rews[:, t] + gamma * nextvalues * \
            nextnonterminal - buf.values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma *\
            gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + buf.values
    return returns.view(-1), advantages.view(-1)
