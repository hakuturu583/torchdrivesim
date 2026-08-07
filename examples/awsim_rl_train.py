"""
PPO training on the AWSIM/Autoware map for TorchDriveSim.

A compact, dependency-light PPO (actor-critic MLP, GAE, clipped surrogate) with
hyper-parameters taken from the reference RL setup in
https://github.com/Emerge-Lab/Adaptive_Driving_Agent (PufferDrive):
    gamma=0.98, gae_lambda=0.95, clip=0.2, ent_coef=0.005, vf_coef=2.0, lr=3e-3.

The num_agents vehicles in one AWSIM scene are treated as parallel experience
streams sharing a single policy (as in GPUDrive / PufferDrive).

Extra deps: pip install imageio-ffmpeg   (mp4 output); matplotlib is optional
(reward curve). The environment/PPO themselves need only torch + torchdrivesim.

Smoke test (CPU, a handful of updates, saves a rollout video):
    python examples/awsim_rl_train.py map_path=sample_map.osm smoke_test=true

Full training (do this on a GPU):
    python examples/awsim_rl_train.py \
        map_path=nishishinjuku_autoware_map/lanelet2_map.osm \
        device=cuda num_agents=32 updates=2000 rollout_steps=256
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from awsim_rl_env import AWSIMDrivingEnv
from awsim_hetero_rl_env import AWSIMHeteroDrivingEnv
from awsim_lanelet2_traffic import save_video


@dataclass
class PPOConfig:
    map_path: str = "sample_map.osm"
    save_dir: str = "./awsim_rl_output"
    device: str = "cpu"
    num_agents: int = 8
    max_steps: int = 80
    dt: float = 0.1
    # PPO (reference PufferDrive values)
    updates: int = 300
    rollout_steps: int = 256
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 2.0
    lr: float = 3e-3
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatches: int = 4
    hidden: int = 128
    seed: int = 42
    smoke_test: bool = False
    hetero: bool = False  # train mixed vehicle/motorcycle/cyclist/pedestrian agents
    checkpoint_every: int = 0  # save policy_<update>.pt every N updates (0 = only final)
    resume: str = ""           # path to a policy .pt to warm-start from
    wandb: bool = False        # log metrics to Weights & Biases
    wandb_project: str = "awsim-rl"
    wandb_run: str = ""        # optional run name
    # rollout video camera: whole map by default, which is unreadable on a 1 km map
    video_follow: int = 0      # agent index to centre on (-1 = frame the whole map)
    video_fov: float = 120.0   # metres across the frame (ignored when video_follow < 0)
    # reward weights (see AWSIMDrivingEnv.W_*); progress is in metres and dominates by default
    # rolling goals: on arrival the next goal is placed goal_dist further along the route
    rolling_goals: bool = True
    goal_dist: float = -1.0    # base env only, metres; <0 keeps the env default
    w_progress: float = 1.0
    w_goal: float = 1.0
    w_offroad: float = 0.6
    w_collision: float = 6.0    # charged on the rise in overlap, ~9.6x smaller than the level
    # traffic-rule penalties (all three are also present in the observation)
    w_redlight: float = 0.5
    w_wrongway: float = 0.2
    w_speeding: float = 0.6
    w_yield: float = 0.5
    n_parked: int = 16         # stationary cars on straight lanelets (hetero env only)
    w_proximity: float = 0.3
    w_lanechange: float = 0.0        # crossing a dashed line onto a neighbouring lane
    w_solidcross: float = 0.0        # crossing a line the map says may not be crossed
    w_collision_level: float = 0.0   # kept alongside the rise; 0 reproduces hit300
    value_norm: bool = False         # standardise the value targets (MAPPO)
    penalty_scale: float = 1.0       # one dial on every penalty, see below
    max_steer_scale: float = 1.0     # scales every type's geometric steering limit
    lat_accel_max: float = 4.0       # m/s^2; the speed-dependent steering cap
    w_lane: float = 0.4         # lateral offset from the lane centre (see W_LANE)
    ttc_threshold: float = 3.0
    # Rule penalties ramp in over the first `penalty_warmup` updates, from
    # `penalty_warmup_start` of their value to full. Applied from the start they punish
    # driving before the policy can drive, and standing still - a safe zero - wins:
    # vehicles converged to 3 km/h and zero goals with every rule metric looking ideal.
    penalty_warmup: int = 200
    penalty_warmup_start: float = 0.1


class ActorCritic(nn.Module):
    """GPUDrive / Nocturne-style actor-critic: separate ego, neighbour and
    road-graph encoders, with the neighbour and road sets max-pooled
    (permutation-invariant deep-sets) so variable, unordered sets are handled."""

    def __init__(self, ego_dim, max_partners, partner_features, max_road, road_features,
                 act_dim, hidden=128, num_types=1, type_slice=None):
        super().__init__()
        self.ego_dim = ego_dim
        self.max_partners, self.partner_features = max_partners, partner_features
        self.max_road, self.road_features = max_road, road_features
        self.partner_dim = max_partners * partner_features
        self.num_types, self.type_slice = num_types, type_slice
        self.ego_enc = nn.Sequential(nn.Linear(ego_dim, hidden), nn.Tanh())
        self.partner_enc = nn.Sequential(nn.Linear(partner_features, hidden), nn.Tanh())
        self.road_enc = nn.Sequential(nn.Linear(road_features, hidden), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh())
        # One actor head (mean + log_std) per agent type; the value head is shared,
        # since ego/partner/road perception is shared and only the action semantics
        # differ per type (car steering vs. omnidirectional pedestrian, etc.).
        self.mean = nn.ModuleList([nn.Linear(hidden, act_dim) for _ in range(num_types)])
        self.log_std = nn.Parameter(-0.5 * torch.ones(num_types, act_dim))
        self.value = nn.Linear(hidden, 1)

    # Entropy of a Gaussian is a constant plus sum(log_std), so the `- ent_coef * entropy`
    # term rewards growing log_std at a fixed rate; with log_std a free parameter and the
    # policy-gradient signal weak, it drifts up without bound. Actions are clipped to
    # [-1, 1] by the env, so once std is around 1 the rollouts are little more than noise
    # and learning stalls while the deterministic (mean) policy still evaluates well.
    LOG_STD_MIN, LOG_STD_MAX = -2.5, 0.0   # std in [0.08, 1.0]

    def _trunk(self, obs):
        B = obs.shape[0]
        ego = obs[:, :self.ego_dim]
        partners = obs[:, self.ego_dim:self.ego_dim + self.partner_dim].view(
            B, self.max_partners, self.partner_features)
        road = obs[:, self.ego_dim + self.partner_dim:].view(B, self.max_road, self.road_features)
        e = self.ego_enc(ego)
        p = self.partner_enc(partners).max(dim=1).values   # deep-sets max-pool over neighbours
        r = self.road_enc(road).max(dim=1).values           # deep-sets max-pool over road points
        return self.trunk(torch.cat([e, p, r], dim=-1))

    def forward(self, obs):
        h = self._trunk(obs)
        log_std_all = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        if self.num_types == 1:
            mean = self.mean[0](h)
            log_std = log_std_all[0].expand_as(mean)
        else:  # select each sample's per-type head using its ego type one-hot
            tidx = obs[:, self.type_slice[0]:self.type_slice[1]].argmax(dim=1)  # [B]
            means = torch.stack([head(h) for head in self.mean], dim=1)         # [B, T, act]
            mean = means[torch.arange(h.shape[0], device=h.device), tidx]
            log_std = log_std_all[tidx]
        return mean, log_std, self.value(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        mean, log_std, value = self(obs)
        std = log_std.exp()
        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)
        logp = self._logp(mean, std, action)
        return action, logp, value

    @staticmethod
    def _logp(mean, std, action):
        var = std.pow(2)
        return (-0.5 * (((action - mean) ** 2) / var + 2 * std.log() + np.log(2 * np.pi))).sum(-1)

    def evaluate(self, obs, action):
        mean, log_std, value = self(obs)
        std = log_std.exp()
        logp = self._logp(mean, std, action)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + log_std).sum(-1)
        return logp, entropy, value


class ValueNorm:
    """Running standardisation of the value targets.

    MAPPO (arXiv:2103.01955) finds this the most influential of its five factors and
    reports it "never hurts training". It matters more here than in a single-task
    setting because the return is assembled from thirteen terms on wildly different
    scales - a +5 goal spike against continuous penalties of 0.02 - so the regression
    target's scale moves as the policy changes which terms it collects. The critic
    predicts in normalised space; GAE needs real units, so values are denormalised on
    the way out and targets normalised on the way in.
    """

    def __init__(self, device):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(), x.var(unbiased=False), x.numel()
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = ((self.var * self.count + bv * bc
                     + delta.pow(2) * self.count * bc / tot) / tot)
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def denormalize(self, x):
        return x * (self.var.sqrt() + 1e-8) + self.mean


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T, A = rewards.shape
    adv = torch.zeros(T, A, device=rewards.device)
    last_gae = torch.zeros(A, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


PENALTIES = ('offroad', 'collision', 'collision_level', 'redlight', 'wrongway',
             'speeding', 'yield', 'proximity', 'lane', 'lanechange', 'solidcross')


def train(cfg: PPOConfig):
    if cfg.penalty_scale != 1.0:
        # Measured on the trained policy, the penalties come to 0.111 per step against
        # 0.095 of progress and goal: driving is negative-expected-value, and the policy
        # answers by driving as little as it can. Removing the lane term alone (18.9% of
        # the penalty budget) took speed/limit 0.52 -> 0.58 and the return positive, so
        # this is one dial on all of them rather than another hunt for the guilty term.
        for k in PENALTIES:
            setattr(cfg, f'w_{k}', getattr(cfg, f'w_{k}') * cfg.penalty_scale)
        print(f"[cfg] penalties scaled by {cfg.penalty_scale}", flush=True)
    if cfg.smoke_test:
        cfg.updates, cfg.rollout_steps = 5, 128
        cfg.num_agents = 12 if cfg.hetero else 6
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device

    env_cls = AWSIMHeteroDrivingEnv if cfg.hetero else AWSIMDrivingEnv
    env = env_cls(cfg.map_path, num_agents=cfg.num_agents, max_steps=cfg.max_steps,
                  dt=cfg.dt, device=dev, seed=cfg.seed, rolling_goals=cfg.rolling_goals,
                  goal_dist=(cfg.goal_dist if cfg.goal_dist > 0 else None),
                  w_progress=cfg.w_progress, w_goal=cfg.w_goal,
                  w_offroad=cfg.w_offroad, w_collision=cfg.w_collision,
                  w_redlight=cfg.w_redlight, w_wrongway=cfg.w_wrongway,
                  w_speeding=cfg.w_speeding, w_yield=cfg.w_yield,
                  w_proximity=cfg.w_proximity, ttc_threshold=cfg.ttc_threshold,
                  w_lane=cfg.w_lane, w_collision_level=cfg.w_collision_level,
                  lat_accel_max=cfg.lat_accel_max, max_steer_scale=cfg.max_steer_scale,
                  w_lanechange=cfg.w_lanechange, w_solidcross=cfg.w_solidcross,
                  **({'n_parked': cfg.n_parked} if cfg.hetero else {}))
    A, od, ad = cfg.num_agents, env.OBS_DIM, env.ACT_DIM
    net = ActorCritic(env.EGO_DIM, env.MAX_PARTNERS, env.PARTNER_FEATURES,
                      env.MAX_ROAD, env.ROAD_FEATURES, ad, cfg.hidden,
                      num_types=env.NUM_TYPES, type_slice=env.TYPE_ONEHOT_SLICE).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-8)
    vnorm = ValueNorm(dev) if cfg.value_norm else None
    if cfg.resume:
        net.load_state_dict(torch.load(cfg.resume, map_location=dev))
        print(f"[resume] loaded policy from {cfg.resume}")

    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project, name=(cfg.wandb_run or None),
                         config=OmegaConf.to_container(cfg, resolve=True))

    obs = env.reset()
    ep_return = torch.zeros(A, device=dev)
    history, sweep_hist = [], []

    for update in range(cfg.updates):
        b_obs = torch.zeros(cfg.rollout_steps, A, od, device=dev)
        b_act = torch.zeros(cfg.rollout_steps, A, ad, device=dev)
        b_logp = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_val = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_rew = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_done = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_active = torch.zeros(cfg.rollout_steps, A, device=dev)
        completed_returns = []
        if cfg.penalty_warmup > 0:
            f = min(1.0, cfg.penalty_warmup_start + (1 - cfg.penalty_warmup_start)
                    * update / cfg.penalty_warmup)
            for k in ('offroad', 'collision', 'redlight', 'wrongway', 'speeding', 'yield',
                      'proximity', 'lane', 'collision_level', 'lanechange', 'solidcross'):
                setattr(env, f'w_{k}', getattr(cfg, f'w_{k}') * f)
        # `info` is a snapshot of the step it came from: `reached` accumulates over an
        # episode and resets with it, so reading it off the last rollout step samples a
        # different point of the episode every update (rollout_steps % max_steps steps
        # further in each time) and swings wildly for no reason related to learning.
        # Goal-reaching is therefore collected at episode boundaries, and the per-step
        # infraction rates are averaged over the whole rollout.
        ep_reached, ep_reached_type, ep_goals, ep_lost = [], {}, [], []
        step_speed = []
        step_coll, step_off, step_rule = [], [], {'redlight': [], 'wrongway': [], 'speeding': [], 'speed_excess': [], 'failtoyield': [], 'proximity': [], 'lane': [], 'contact': [], 'lanechange': [], 'solidcross': []}

        for t in range(cfg.rollout_steps):
            with torch.no_grad():
                action, logp, value = net.act(obs)
            next_obs, reward, done, info = env.step(action)
            b_obs[t], b_act[t], b_logp[t] = obs, action, logp
            b_val[t] = vnorm.denormalize(value) if vnorm else value
            b_rew[t], b_done[t] = reward, done.float()
            b_active[t] = info['active'].float()
            ep_return += reward
            obs = next_obs
            step_coll.append(info['collision'])
            step_off.append(info['offroad'])
            step_speed.append(info['speed_ratio'])
            for k in step_rule:
                step_rule[k].append(info[k])
            if info['episode_end']:  # episode boundary -> log and reset (no device read)
                completed_returns.append(ep_return.mean())
                ep_reached.append(info['reached'])
                ep_goals.append(info['goals'])
                ep_lost.append(info['lost'])
                for k, v in info.items():
                    if k.startswith(('reached_', 'goals_', 'speed_')):
                        ep_reached_type.setdefault(k, []).append(v)
                ep_return = torch.zeros(A, device=dev)
                obs = env.reset()

        with torch.no_grad():
            last_value = net(obs)[2]
            if vnorm:
                last_value = vnorm.denormalize(last_value)
        adv, returns = compute_gae(b_rew, b_val, b_done, last_value, cfg.gamma, cfg.gae_lambda)
        if vnorm:
            # update on this batch's targets, then regress against them standardised
            vnorm.update(returns)
            returns = vnorm.normalize(returns)

        f_obs = b_obs.reshape(-1, od)
        f_act = b_act.reshape(-1, ad)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)
        # Only train on transitions before each agent finished (post-goal steps excluded).
        active_idx = b_active.reshape(-1).bool().nonzero(as_tuple=True)[0].cpu().numpy()
        if active_idx.size == 0:
            active_idx = np.arange(f_obs.shape[0])
        f_adv = (f_adv - f_adv[active_idx].mean()) / (f_adv[active_idx].std() + 1e-8)

        N = active_idx.size
        mb = max(1, N // cfg.minibatches)
        last_stats = (0.0, 0.0, 0.0)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(active_idx)
            for start in range(0, N, mb):
                j = torch.as_tensor(active_idx[start:start + mb], device=dev)
                new_logp, entropy, value = net.evaluate(f_obs[j], f_act[j])
                ratio = (new_logp - f_logp[j]).exp()
                pg1 = -f_adv[j] * ratio
                pg2 = -f_adv[j] * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value - f_ret[j]).pow(2).mean()
                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                last_stats = (pg_loss.item(), v_loss.item(), ent.item())

        # one device read per rollout instead of per step: stacking keeps everything on
        # the GPU until here, which is what the metrics used to cost 42 ms a step for
        stack = lambda xs: float(torch.stack([torch.as_tensor(x, device=dev) for x in xs]).mean())
        mean_ret = stack(completed_returns) if completed_returns else float(b_rew.sum(0).mean())
        history.append(mean_ret)
        # falls back to the last snapshot only when the rollout spans no full episode
        mean_reached = stack(ep_reached) if ep_reached else float(info['reached'])
        mean_goals = stack(ep_goals) if ep_goals else float(info['goals'])
        mean_lost = float(np.mean(ep_lost)) if ep_lost else float(info['lost'])
        mean_speed = stack(step_speed)
        mean_coll, mean_off = stack(step_coll), stack(step_off)
        mean_rule = {k: stack(v) for k, v in step_rule.items()}
        sweep_hist.append((mean_goals, mean_coll, mean_rule['failtoyield'], mean_speed))
        reached_type = {k: stack(v) for k, v in ep_reached_type.items()} or \
                       {k: float(v) for k, v in info.items() if k.startswith('reached_')}
        print(f"upd {update+1:4d}/{cfg.updates} | return {mean_ret:8.3f} | "
              f"goals {mean_goals:5.2f} reached {mean_reached:.2f} coll {mean_coll:.2f} "
              f"v/lim {mean_speed:.2f} off {mean_off:.2f} lost {mean_lost:4.0f} red {mean_rule['redlight']:.2f} "
              f"prox {mean_rule['proximity']:.3f} wrong {mean_rule['wrongway']:.2f} spd {mean_rule['speeding']:.2f}"
              f"/{mean_rule['speed_excess']:.2f} yld {mean_rule['failtoyield']:.3f} "
              f"lane {mean_rule['lane']:.3f} hit {mean_rule['contact']:.4f} "
              f"chg {mean_rule['lanechange']:.4f}/{mean_rule['solidcross']:.4f} | "
              f"pg {last_stats[0]:.3f} vf {last_stats[1]:.3f} ent {last_stats[2]:.3f}", flush=True)
        if cfg.checkpoint_every and (update + 1) % cfg.checkpoint_every == 0:
            torch.save(net.state_dict(), os.path.join(cfg.save_dir, f"policy_{update + 1}.pt"))
        if run is not None:
            metrics = {"return": mean_ret, "reached": mean_reached, "goals": mean_goals, "lost": mean_lost, "speed_ratio": mean_speed,
                       "collision": mean_coll, "offroad": mean_off,
                       "loss/policy": last_stats[0], "loss/value": last_stats[1],
                       "entropy": last_stats[2],
                       **{f"rule/{k}": v for k, v in mean_rule.items()}}
            for k, v in reached_type.items():
                pre, name = k.split('_', 1)
                metrics[f'{pre}/{name}'] = v
            run.log(metrics, step=update)

    # save reward curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(history)
        plt.xlabel("update"); plt.ylabel("mean episode return"); plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_dir, "reward_curve.png"))
    except Exception as exc:
        print(f"[plot] skipped: {exc}")

    # Sweep objective. Collision and yield rates alone are maximised by standing still -
    # that failure has already happened once here - so the score has to carry task
    # performance too. `goals` does that: when the wheeled agents parked it fell to 2.68
    # (pedestrians only) against 3.89 for a policy that drives.
    tail = sweep_hist[-20:] or [(0.0, 0.0, 0.0, 0.0)]
    g, c, y, v = (float(np.mean([t[i] for t in tail])) for i in range(4))
    score = g - 10.0 * c - 10.0 * y
    print(f"[sweep] score {score:.3f} = goals {g:.3f} - 10*coll {c:.4f} - 10*yield {y:.4f}"
          f"  (v/lim {v:.2f}, mean of last {len(tail)} updates)", flush=True)
    if run is not None:
        run.summary.update({"sweep/score": score, "sweep/goals": g, "sweep/collision": c,
                            "sweep/failtoyield": y, "sweep/speed_ratio": v})

    torch.save(net.state_dict(), os.path.join(cfg.save_dir, "policy.pt"))

    # greedy evaluation rollout -> video
    obs = env.reset()
    cam = dict(follow=cfg.video_follow, fov=cfg.video_fov) if cfg.video_follow >= 0 else {}
    frames = [env.render_frame(**cam)]
    for _ in range(cfg.max_steps):
        with torch.no_grad():
            action, _, _ = net.act(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        frames.append(env.render_frame(**cam))
        if info['episode_end']:
            break
    video = save_video(frames, cfg.save_dir, "awsim_rl_rollout", cfg.dt, "mp4")
    print(f"[done] policy -> {os.path.join(cfg.save_dir, 'policy.pt')}")
    print(f"[done] rollout video -> {video}  (final goals/agent {float(info['goals']):.2f}, "
          f"reached {float(info['reached']):.2f})")
    per_type = {k: v for k, v in info.items() if k.startswith('goals_')}
    if per_type:
        print("[done] per-type goals/agent: " + ", ".join(f"{k[6:]} {float(v):.2f}" for k, v in per_type.items()))
    per_speed = {k: v for k, v in info.items() if k.startswith('speed_')}
    if per_speed:
        print("[done] per-type v/limit: " + ", ".join(f"{k[6:]} {float(v):.2f}" for k, v in per_speed.items()))

    if run is not None:
        import wandb
        log = {"final/reached": float(info["reached"]), "rollout": wandb.Video(video)}
        curve = os.path.join(cfg.save_dir, "reward_curve.png")
        if os.path.exists(curve):
            log["reward_curve"] = wandb.Image(curve)
        run.log(log)
        run.finish()


if __name__ == '__main__':
    cli_cfg: PPOConfig = OmegaConf.structured(PPOConfig(**OmegaConf.from_dotlist(sys.argv[1:])))
    train(cli_cfg)  # type: ignore
