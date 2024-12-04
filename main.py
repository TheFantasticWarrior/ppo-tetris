import core
import mp
import model
import args
import torch
import numpy as np
import torch.nn as nn


def main():
    env = mp.manager(args.nenvs, args.seed, args.render)

    obs = env.reset()
    done = np.zeros(args.nenvs, dtype=bool)
    mem0 = torch.zeros((args.nenvs, 4, args.d_model),
                       device=torch.device("cuda"))
    mem1 = torch.zeros((args.nenvs, 4, args.d_model),
                       device=torch.device("cuda"))
    buf = core.Buffer(args.nsteps, args.nenvs)
    main_model = model.Model().to(torch.device("cuda"))
    params = main_model.parameters()
    optimizer = torch.optim.AdamW(params, args.lr)
    if args.load:
        checkpoint = torch.load(args.path, weights_only=True)
        if args.partial:
            ms = checkpoint['model_state_dict']
            model_dict = main_model.state_dict()
            pretrained_dict = {k: v for k, v in ms.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            main_model.load_state_dict(model_dict)

        else:
            main_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    for i in range(args.iterations):
        for step in range(args.nsteps):
            buf.record_obs(step, obs)
            done = torch.tensor(done)
            buf.done[step] = done

            buf.memory[0, step][~done] = mem0
            buf.memory[1, step][~done] = mem1
            with torch.no_grad():
                p1_logit, buf.values[0, step] = main_model(
                    [ob[0, step] for ob in buf.obs])
                p2_logit, buf.values[1, step] = main_model(
                    [ob[1, step] for ob in buf.obs])

            buf.actions[0, step], buf.logprob[0, step] = core.sample(p1_logit)
            buf.actions[1, step], buf.logprob[1, step] = core.sample(p2_logit)

            env.send(buf.actions[:, step].int().cpu().numpy().T)

            obs, done, rew = env.recv()
            buf.rews[:, step] = torch.tensor(rew)
        count = max(1, env.count)
        print(f"iteration {i}, {env.count} games, {env.lines/count:.2f} lines, {env.atk /
              count:.2f} atk, {buf.rews.sum().cpu().numpy():.4f} rewards total")
        env.reset_data()
        with torch.no_grad():
            _, next_val1 = main_model(
                [torch.tensor(ob[0], device=torch.device("cuda")) for ob in obs])
            _, next_val2 = main_model(
                [torch.tensor(ob[1], device=torch.device("cuda")) for ob in obs])

            returns, advs = core.calc_gae(
                buf, torch.stack((next_val1, next_val2)),
                torch.tensor(done, device=torch.device("cuda")))
        print("return max {:.4f} min {:.4f}, mean{:.3f}".format(
            returns.max(), returns.min(), returns.mean()))
        flat_obs, flat_buf = buf.flatten()
        for _ in range(args.nepoch):
            arr = np.arange(args.nenvs*args.nsteps * 2)
            order = np.delete(arr, [0, args.nenvs*args.nsteps])
            np.random.shuffle(order)
            start = 0
            for _ in range(args.nminibatch):
                sli = order[start:start+args.samplesperbatch]
                start += args.samplesperbatch

                acs, log_prob, vals = map(
                    lambda x: x[sli].detach(), flat_buf[:3])

                logits, new_values = main_model([o[sli] for o in flat_obs])
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

        val = flat_buf[2]
        ev = 1-(returns-val).var() / \
            returns.var() if returns.var() != 0 else np.nan
        print(f"{pg_loss=:.4f} {v_loss=:.4f} {entropy_loss=:.4f} {ev=:.2f}")
        if (i % 5) == 0 and i > 1:
            torch.save({
                'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.path)
            print("saved")


if __name__ == "__main__":
    main()
