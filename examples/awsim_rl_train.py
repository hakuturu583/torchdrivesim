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


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        mean = self.mean(h)
        return mean, self.log_std.expand_as(mean), self.value(h).squeeze(-1)

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


def train(cfg: PPOConfig):
    if cfg.smoke_test:
        cfg.updates, cfg.rollout_steps = 5, 128
        cfg.num_agents = 12 if cfg.hetero else 6
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device

    env_cls = AWSIMHeteroDrivingEnv if cfg.hetero else AWSIMDrivingEnv
    env = env_cls(cfg.map_path, num_agents=cfg.num_agents, max_steps=cfg.max_steps,
                  dt=cfg.dt, device=dev, seed=cfg.seed)
    A, od, ad = cfg.num_agents, env.OBS_DIM, env.ACT_DIM
    net = ActorCritic(od, ad, cfg.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-8)

    obs = env.reset()
    ep_return = torch.zeros(A, device=dev)
    history = []

    for update in range(cfg.updates):
        b_obs = torch.zeros(cfg.rollout_steps, A, od, device=dev)
        b_act = torch.zeros(cfg.rollout_steps, A, ad, device=dev)
        b_logp = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_val = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_rew = torch.zeros(cfg.rollout_steps, A, device=dev)
        b_done = torch.zeros(cfg.rollout_steps, A, device=dev)
        completed_returns = []

        for t in range(cfg.rollout_steps):
            with torch.no_grad():
                action, logp, value = net.act(obs)
            next_obs, reward, done, info = env.step(action)
            b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, action, logp, value
            b_rew[t], b_done[t] = reward, done.float()
            ep_return += reward
            obs = next_obs
            if bool(done.all()):  # episode boundary -> log and reset
                completed_returns.append(float(ep_return.mean()))
                ep_return = torch.zeros(A, device=dev)
                obs = env.reset()

        with torch.no_grad():
            last_value = net(obs)[2]
        adv, returns = compute_gae(b_rew, b_val, b_done, last_value, cfg.gamma, cfg.gae_lambda)

        f_obs = b_obs.reshape(-1, od)
        f_act = b_act.reshape(-1, ad)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        N = f_obs.shape[0]
        mb = max(1, N // cfg.minibatches)
        idx = np.arange(N)
        last_stats = (0.0, 0.0, 0.0)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb):
                j = idx[start:start + mb]
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

        mean_ret = np.mean(completed_returns) if completed_returns else float(b_rew.sum(0).mean())
        history.append(mean_ret)
        print(f"upd {update+1:4d}/{cfg.updates} | return {mean_ret:8.3f} | "
              f"reached {info['reached']:.2f} coll {info['collision']:.2f} off {info['offroad']:.2f} | "
              f"pg {last_stats[0]:.3f} vf {last_stats[1]:.3f} ent {last_stats[2]:.3f}")

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

    torch.save(net.state_dict(), os.path.join(cfg.save_dir, "policy.pt"))

    # greedy evaluation rollout -> video
    obs = env.reset()
    frames = [env.render_frame()]
    for _ in range(cfg.max_steps):
        with torch.no_grad():
            action, _, _ = net.act(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        frames.append(env.render_frame())
        if bool(done.all()):
            break
    video = save_video(frames, cfg.save_dir, "awsim_rl_rollout", cfg.dt, "mp4")
    print(f"[done] policy -> {os.path.join(cfg.save_dir, 'policy.pt')}")
    print(f"[done] rollout video -> {video}  (final reached {info['reached']:.2f})")
    per_type = {k: v for k, v in info.items() if k.startswith('reached_')}
    if per_type:
        print("[done] per-type reached: " + ", ".join(f"{k[8:]} {v:.2f}" for k, v in per_type.items()))


if __name__ == '__main__':
    cli_cfg: PPOConfig = OmegaConf.structured(PPOConfig(**OmegaConf.from_dotlist(sys.argv[1:])))
    train(cli_cfg)  # type: ignore
