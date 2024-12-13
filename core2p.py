import torch
import args
import copy
import numpy as np


class ModelPool:
    def __init__(self, model):
        model.cpu()
        self.models = [model.state_dict()]
        self.scores = [1]
        self.sumexpscore = np.exp(1)
        self.active_models = [copy.deepcopy(model) for i in range(args.nopps)]
        model.cuda()
        self.active_idx = np.zeros(args.nopps, dtype=int)

    def update(self, model):
        model.cpu()
        self.models.append(model.state_dict())
        model.cuda()
        new_score = max(self.scores) if len(self.scores) else 1
        # print(self.scores)
        self.scores.append(new_score)
        # if len(self.models) > 20:
        #     idx = self.scores.index(min(self.scores))
        #     del self.scores[idx]
        #     del self.models[idx]

    def sample(self, dones, win):
        # if done update score and sample from model pool
        if not dones.any():
            return
        remove = []
        for i in reversed(range(args.nopps)):
            if win[i] or not dones[i]:
                continue
            score = self.scores[self.active_idx[i]]
            prob = np.exp(score)/self.sumexpscore

            if prob < 0.001:
                remove.append(self.active_idx[i])

            else:
                new_score = score-0.01/prob/len(self.models)
                self.scores[self.active_idx[i]] = new_score
        if remove:
            for i in reversed(sorted(remove)):
                self.active_idx[self.active_idx > i] -= 1
                del self.scores[i]
                del self.models[i]
        sum_dones = dones.sum().item()
        new_probs = np.exp(self.scores)
        self.sumexpscore = new_probs.sum()
        new_idx = np.random.choice(
            len(self.models), size=sum_dones, p=new_probs/self.sumexpscore)
        n = 0
        for i in range(sum_dones):
            if dones[i]:
                if self.active_idx[i] != new_idx[n]:
                    self.active_models[i].load_state_dict(
                        self.models[new_idx[n]])
                    self.active_idx[i] = new_idx[n]
                n += 1

    def forward(self, x, y, mem, done):
        x = zip(*[model([o[i:i+1] for o in x],
                        [o[i:i+1] for o in y],
                        mem[i:i+1], done[i:i+1]) for i, model in
                  enumerate(self.active_models)])
        return x


class Buffer:
    def __init__(self, nsteps, nenvs):

        self.obs = [torch.zeros((nsteps, nenvs, 4), dtype=torch.float32).to(torch.device("cuda")),
                    torch.zeros((nsteps, nenvs, 7), dtype=torch.int).to(
            torch.device("cuda")),
            torch.zeros((nsteps, nenvs, 7), dtype=torch.int).to(
            torch.device("cuda")),
            torch.zeros((nsteps, nenvs, 22, 12)).to(torch.device("cuda"),
                                                    dtype=torch.float32),
            torch.zeros((nsteps, nenvs, 10), dtype=torch.float32).to(
            torch.device("cuda"))
        ]
        self.memory = torch.zeros(
            (nsteps, nenvs, 6, args.d_model), dtype=torch.float32).to(torch.device("cuda"))
        self.done = torch.zeros((nsteps, nenvs), dtype=bool).to(
            torch.device("cuda"))
        self.rews = torch.zeros((nsteps, nenvs)).to(torch.device("cuda"))
        self.actions = torch.zeros((nsteps, nenvs)).to(torch.device("cuda"))
        self.logprob = torch.zeros((nsteps, nenvs)).to(torch.device("cuda"))
        self.values = torch.zeros((nsteps, nenvs)).to(torch.device("cuda"))
        self.size = nsteps*nenvs
        self.nenvs = nenvs

    def record_obs(self, step, obs):
        for i in range(5):
            self.obs[i][step] = obs[i]

    def record_others(self, step, *others):
        for storage, value in zip(
                (self.rews, self.values, self.actions, self.logprob), others):
            storage[step] = value[:self.nenvs]

    def flat_obs(self, step):
        return [x.view(-1, *x.shape[2:])[step] for x in self.obs]


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
    num_steps = advantages.size(0)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = ~last_done
            nextvalues = next_vals
        else:
            nextnonterminal = ~buf.done[t + 1]
            nextvalues = buf.values[t + 1]
        delta = buf.rews[t] + gamma * nextvalues *\
            nextnonterminal - buf.values[t]
        advantages[t] = lastgaelam = delta + gamma *\
            gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + buf.values
    return returns.view(-1), advantages.view(-1)


def split_obs(obs):
    inputs = [[] for i in range(4)]
    for i in range(5):
        x = obs[i]
        inputs[0].append(torch.tensor(
            x.reshape(-1, *x.shape[2:])[:-args.nopps],
            device=torch.device("cuda")))
        inputs[1].append(torch.tensor(np.concatenate(
            (x[1], x[0, :-args.nopps]), 0), device=torch.device("cuda")))
        inputs[2].append(torch.tensor(
            x[1, -args.nopps:], device=torch.device("cuda")))
        inputs[3].append(torch.tensor(
            x[0, -args.nopps:], device=torch.device("cuda")))
    return inputs


def warmup(current_step: int):
    if current_step < args.warmup_steps:
        return float((1+current_step) / args.warmup_steps)
    else:
        return max(0.0, float(args.training_steps - current_step) /
                   float(max(1, args.training_steps - args.warmup_steps)))


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"[WARNING] No gradient for {name}")
        else:
            grad_norm = param.grad.norm().item()
            print(f"[INFO] Gradient norm for {name}: {grad_norm:.4e}")


def debug_actions(logits):
    with torch.no_grad():
        action_probs = torch.softmax(logits, dim=-1)
        variances = action_probs.var(dim=0).mean().item()
        print(f"[INFO] Mean action probability variance: {
              variances:.4e}")

        max_probs = action_probs.max(dim=-1).values
        print(f"[INFO] Mean max action probability: {
              max_probs.mean().item():.4e}")
