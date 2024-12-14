def load(path, model, optimizer, return_pool=False):
    checkpoint = torch.load(path, weights_only=False)
    reset_layers = []
    # reset_layers = ['model.value.weight', 'model.value.bias',
    # reset_layers = ['model.policy.layers.1.weight',
    # 'model.policy.layers.1.bias',
    #                 'value.layers.1.weight', 'value.layers.1.bias',
    #                 'macro_value.weight', 'macro_value.bias']
    newstatedict = checkpoint['model2_state_dict']
    if reset_layers:
        oldstatedict = model.state_dict()
        for key in reset_layers:
            newstatedict[key] = oldstatedict[key]
    if args.partial:
        ms = checkpoint['model2_state_dict']
        model_dict = model.model.state_dict()
        pretrained_dict = {k: v for k, v in ms.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.model.load_state_dict(model_dict)
        # model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(newstatedict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if return_pool:
        pool = checkpoint['pool']
        print(len(pool.scores))
        return pool


def main(lr):
    no_win = True

    torch.autograd.set_detect_anomaly(True)
    env = mp.manager(args.nenvs, args.seed, args.render)

    obs = env.reset()
    nenvs_self = args.nenvs*2 - args.nopps
    done = torch.ones(nenvs_self, dtype=bool)
    buf = core.Buffer(args.nsteps, nenvs_self)
    buf_opp = core.Buffer(args.nsteps, nenvs_self)
    mem0, mem1 = (lambda x: (x[:nenvs_self], x[:args.nopps]))(
        buf.memory[0].clone())
    model = models.SModel2().to(torch.device("cuda"))
    # model = models.MacroModel().to(torch.device("cuda"))
    pool = core.ModelPool(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=core.warmup)
    data = {'step': [],
            'entropy': [],
            'pg_loss': [],
            'value_loss': [],
            'lr': []
            }
    iteration_data = {'iteration': [],
                      'returns_mean': [],
                      'total rewards': [],
                      'var rewards': [],
                      'explained variance': [],
                      'lines': [],
                      'atk': [],
                      'filled%': [],
                      }
    if args.load:
        pool = load(args.path, model, optimizer, True)

    for i in range(args.iterations):
        for step in range(args.nsteps):
            buf.done[step] = done

            inputs = core.split_obs(obs)

            buf.record_obs(step, inputs[0])
            buf_opp.record_obs(step, inputs[1])
            with torch.no_grad():
                logits, values, mem0 = model(
                    inputs[0], inputs[1], mem0,
                    buf.done[step])
                logits2, _, mem1 = pool.forward(
                    inputs[2], inputs[3],
                    mem1, done[-args.nopps:])
                logits2 = torch.cat(logits2, 0)
                if mem0 is not None:
                    buf.memory[step] = mem0
                    if mem1 is not None:
                        mem1 = torch.cat(mem1, 0)
            logits = torch.cat((logits, logits2), 0)
            actions, logprobs = core.sample(logits)

            env.send(actions.view(2, -1).int().cpu().numpy().T)

            obs, done, rew = env.recv()
            done = torch.tensor(done, dtype=bool, device=torch.device("cuda"))
            pool.sample(done[-args.nopps:], rew[1, -args.nopps:] == 1)
            buf.record_others(step, torch.tensor(rew).flatten(), values,
                              actions, logprobs)
            done = done.expand(2, -1).reshape(-1)[:nenvs_self]
        count = max(1, env.count)
        sum_rews = buf.rews.mean().item()
        rews_var = buf.rews.var().item()
        kpp = args.nenvs*args.nsteps*2/(buf.actions == 1).sum().item()
        print(f"iteration {i+1}, {kpp=:.2f}, {env.count} deaths, " +
              f"{sum_rews:.4f} rewards total, {rews_var=:.4f}")
        print(f"{env.lines/count:.2f} lines, {env.atk / count:.2f} atk, " +
              f"avg {env.filled_avg/count*100:.1f}% filled")
        # print(buf.actions.cpu().numpy()[0,:,0])
        with torch.no_grad():
            inputs = core.split_obs(obs)
            _, next_val, _ = model(inputs[0], inputs[1], mem0,
                                   done.expand(2, -1).reshape(-1)[:nenvs_self])
            if no_win:
                done = torch.tensor(
                    rew == -1,
                    device=torch.device("cuda")).flatten()[:nenvs_self]
            returns, advs = core.calc_gae(
                buf, next_val, done, no_win)
        for epoch in range(args.nepoch):
            order = np.arange(nenvs_self*args.nsteps)
            np.random.shuffle(order)
            start = 0
            for minibatch in range(args.nminibatch):
                sli = order[start:start+args.samplesperbatch]
                start += args.samplesperbatch

                acs = buf.actions.flatten()[sli]
                log_prob = buf.logprob.flatten()[sli]
                vals = buf.values.flatten()[sli]
                dones = buf.done.flatten().bool()
                if mem0 is None:
                    logits, new_values, _ = model(buf.flat_obs(sli))
                else:
                    mem = model(buf.flat_obs(sli-1),
                                buf_opp.flat_obs(sli-1),
                                buf.memory.view(
                                -1, *buf.memory.shape[2:])[sli-1],
                                dones[sli-1], dones[sli])
                    logits, new_values, _ = model(buf.flat_obs(sli),
                                                  buf_opp.flat_obs(sli),
                                                  mem, dones[sli])
                if args.debug_acs and epoch == 0 and minibatch == 0:
                    core.debug_actions(logits)
                # print(logits.mean(),logits.max(),logits.min())
                entropy, new_log_prob = core.entropy_and_logprob(logits, acs)
                ratio = torch.exp(new_log_prob-log_prob)
                adv = advs[sli]
                adv = (adv-adv.mean())/(adv.std()+1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv *\
                    torch.clamp(ratio, 1 - args.cliprange, 1 + args.cliprange)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_values - returns[sli]) ** 2
                v_clipped = vals + torch.clamp(
                    new_values - vals,
                    -args.cliprange,
                    args.cliprange,
                )
                v_loss_clipped = (v_clipped - returns[sli]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss\
                    + v_loss * args.vf_coef

                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    # core.check_gradients(model)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()

                new_data = {'step':
                            (i*args.nepoch+epoch)*args.nminibatch+minibatch,
                            'entropy': entropy_loss.item(),
                            'pg_loss': pg_loss.item(),
                            'value_loss': v_loss.item(),
                            'lr': lr_scheduler.get_last_lr()
                            }
                for key in data:
                    data[key].append(new_data[key])

        val = buf.values.flatten()
        ev = 1-(returns-val).var() /\
            returns.var() if returns.var() != 0 else np.nan
        new_iteration_data = {'iteration': i,
                              'returns_mean': returns.mean().item(),
                              'var rewards': rews_var,
                              'total rewards': sum_rews,
                              'explained variance': ev.item(),
                              'lines': env.lines/count,
                              'atk': env.atk / count,
                              'filled%': env.filled_avg/count
                              }
        env.reset_data()
        for key in new_iteration_data:
            iteration_data[key].append(new_iteration_data[key])

        if ((i+1) % 10) == 0:
            torch.save({
                'model2_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pool': pool
            }, args.path)
            pd.DataFrame(data).to_feather('losses.feather')
            pd.DataFrame(iteration_data).to_feather('data.feather')
            print("saved")
        if ((i+1) % 10) == 0:
            pool.update(model)
    env.close()


if __name__ == "__main__":
    import args
    import pandas as pd
    import core2p as core
    import mp
    import models
    import torch
    import numpy as np
    import torch.nn as nn
    main(args.lr)
