

def main(lr):
    def warmup(current_step: int):
        if current_step < args.warmup_steps:
            return float((1+current_step) / args.warmup_steps)
        else:
            return max(0.0, float(args.training_steps - current_step) / float(max(1, args.training_steps - args.warmup_steps)))

    def check_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"[WARNING] No gradient for {name}")
            else:
                grad_norm = param.grad.norm().item()
                print(f"[INFO] Gradient norm for {name}: {grad_norm:.4e}")
    import pandas as pd
    import core2p as core
    import mp
    import model
    import args
    import torch
    import numpy as np
    import torch.nn as nn
    torch.autograd.set_detect_anomaly(True)
    env = mp.manager(args.nenvs, args.seed, args.render)

    obs = env.reset()
    done = torch.ones(args.nenvs, dtype=bool)
    buf = core.Buffer(args.nsteps, args.nenvs)
    mem0 = mem1 = buf.memory[0, 0].clone()
    pv_model = model.MacroModel().to(torch.device("cuda"))
    optimizer = torch.optim.AdamW(pv_model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup)
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
        checkpoint = torch.load(args.path, weights_only=True)
        # if "loss" in checkpoint:
        #    loss_l = checkpoint["loss"]

        if args.partial:
            ms = checkpoint['model2_state_dict']
            model_dict = pv_model.model.state_dict()
            pretrained_dict = {k: v for k, v in ms.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            pv_model.model.load_state_dict(model_dict)
            # pv_model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pv_model.load_state_dict(checkpoint['model2_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    for i in range(args.iterations):
        for step in range(args.nsteps):
            buf.record_obs(step, obs)
            buf.done[step] = done

            buf.memory[0, step] = mem0
            buf.memory[1, step] = mem1
            with torch.no_grad():
                (p1_logit, buf.values[0, step], mem0)\
                    = pv_model([ob[0, step] for ob in buf.obs],
                               [ob[1, step] for ob in buf.obs],
                               mem0, done)
                (p2_logit, buf.values[1, step], mem1)\
                    = pv_model([ob[1, step] for ob in buf.obs],
                               [ob[0, step] for ob in buf.obs],
                               mem1, done)
            buf.actions[0, step], buf.logprob[0, step] = core.sample(p1_logit)
            buf.actions[1, step], buf.logprob[1, step] = core.sample(p2_logit)

            env.send(buf.actions[:, step].int().cpu().numpy().T)

            obs, done, rew = env.recv()
            done = torch.tensor(done, device=torch.device("cuda"))
            buf.rews[:, step] = torch.tensor(rew)
        count = max(1, env.count)
        sum_rews = buf.rews[:, ~buf.done.bool()].mean().item()
        rews_var = buf.rews.var().item()
        kpp = args.nenvs*args.nsteps*2/(buf.actions == 1).sum().item()
        print(f"iteration {i}, {kpp=:.2f}, {env.count} deaths, " +
              f"{sum_rews:.4f} rewards total, {rews_var=:.4f}")
        print(f"{env.lines/count:.2f} lines, {env.atk / count:.2f} atk, " +
              f"avg {env.filled_avg/count*100:.1f}% filled")
        # print(buf.actions.cpu().numpy()[0,:,0])
        with torch.no_grad():
            _, next_val1, _\
                = pv_model([ob[0, step] for ob in buf.obs],
                           [ob[1, step] for ob in buf.obs],
                           mem0, done)
            _, next_val2, _\
                = pv_model([ob[1, step] for ob in buf.obs],
                           [ob[0, step] for ob in buf.obs],
                           mem1, done)
            returns, advs = core.calc_gae(
                buf, torch.stack((next_val1, next_val2)), done)
        flat_obs, flat_buf = buf.flatten()
        for epoch in range(args.nepoch):
            arr = np.arange(args.nenvs*args.nsteps * 2)
            order = np.delete(arr, [0, args.nenvs*args.nsteps])
            np.random.shuffle(order)
            start = 0
            for minibatch in range(args.nminibatch):
                sli = order[start:start+args.samplesperbatch]
                start += args.samplesperbatch

                acs, log_prob, vals = map(
                    lambda x: x[sli], flat_buf[:3])
                dones = buf.done.expand(2, -1, -1).flatten().bool()
                mem = pv_model([ob[0][sli-1] for ob in flat_obs],
                               [ob[1][sli-1] for ob in flat_obs],
                               flat_buf[-1][sli-1],
                               dones[sli-1], dones[sli])
                logits, new_values, _ =\
                    pv_model([ob[0][sli] for ob in flat_obs],
                             [ob[1][sli] for ob in flat_obs],
                             mem, dones[sli])
                if args.debug_acs and epoch == 0 and minibatch == 0:
                    with torch.no_grad():
                        action_probs = torch.softmax(logits, dim=-1)
                        variances = action_probs.var(dim=0).mean().item()
                        print(f"[INFO] Mean action probability variance: {
                              variances:.4e}")

                        # Optionally: check if actions are dominated by one policy
                        max_probs = action_probs.max(dim=-1).values
                        print(f"[INFO] Mean max action probability: {
                              max_probs.mean().item():.4e}")
                # print(logits.mean(),logits.max(),logits.min())
                entropy, new_log_prob = core.entropy_and_logprob(logits, acs)
                ratio = torch.exp(new_log_prob-log_prob)
                adv = advs[sli]
                adv = (adv-adv.mean())/(adv.std()+1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * \
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
                    # check_gradients(pv_model)
                    nn.utils.clip_grad_norm_(
                        pv_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()

                new_data = {'step': (i*args.nepoch+epoch)*args.nminibatch+minibatch,
                            'entropy': entropy_loss.item(),
                            'pg_loss': pg_loss.item(),
                            'value_loss': v_loss.item(),
                            'lr': lr_scheduler.get_last_lr()
                            }
                for key in data:
                    data[key].append(new_data[key])

        val = flat_buf[2]
        ev = 1-(returns-val).var() / \
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

        if ((i+1) % 5) == 0:
            torch.save({
                'model2_state_dict': pv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.path)
            pd.DataFrame(data).to_feather('losses.feather')
            pd.DataFrame(iteration_data).to_feather('data.feather')
            print("saved")

    env.close()


if __name__ == "__main__":

    import args
    main(args.lr)
