"""
Smoke tests for the AWSIM / Autoware examples and RL scaffold (examples/awsim_*).

Self-contained: a tiny AWSIM-format (Vector Map Builder dialect) Lanelet2 map is
generated on the fly - two straight `road` lanelets in a chain with local_x/local_y
tags - so the tests need no network and no bundled map. Tests are skipped if the
`lanelet2` bindings are unavailable.
"""
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("lanelet2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

pytestmark = pytest.mark.depends_on_lanelet2


def _tiny_awsim_osm():
    """Two straight road lanelets (0-30 m, 30-60 m) along +x, half-width 1.75 m."""
    left = [(1, 0.0, 1.75), (2, 30.0, 1.75), (3, 60.0, 1.75)]
    right = [(4, 0.0, -1.75), (5, 30.0, -1.75), (6, 60.0, -1.75)]

    def node(nid, lx, ly):
        return (f'  <node id="{nid}" lat="{ly * 9e-6:.8f}" lon="{lx * 9e-6:.8f}">\n'
                f'    <tag k="local_x" v="{lx}"/>\n    <tag k="local_y" v="{ly}"/>\n'
                f'    <tag k="ele" v="0.0"/>\n  </node>')

    def way(wid, refs):
        nds = "".join(f'    <nd ref="{r}"/>\n' for r in refs)
        return (f'  <way id="{wid}">\n{nds}    <tag k="type" v="line_thin"/>\n'
                f'    <tag k="subtype" v="solid"/>\n  </way>')

    def lanelet(rid, lw, rw):
        return (f'  <relation id="{rid}">\n'
                f'    <member type="way" role="left" ref="{lw}"/>\n'
                f'    <member type="way" role="right" ref="{rw}"/>\n'
                f'    <tag k="type" v="lanelet"/>\n    <tag k="subtype" v="road"/>\n'
                f'    <tag k="one_way" v="yes"/>\n    <tag k="location" v="urban"/>\n'
                f'    <tag k="participant:vehicle" v="yes"/>\n  </relation>')

    parts = [node(n, lx, ly) for n, lx, ly in left + right]
    parts += [way(100, [1, 2]), way(101, [4, 5]), way(102, [2, 3]), way(103, [5, 6])]
    parts += [lanelet(200, 100, 101), lanelet(201, 102, 103)]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<osm generator="VMB">\n'
            '  <MetaInfo format_version="1" map_version="1"/>\n'
            + "\n".join(parts) + "\n</osm>\n")


@pytest.fixture(scope="module")
def tiny_map(tmp_path_factory):
    path = tmp_path_factory.mktemp("awsim") / "tiny_map.osm"
    path.write_text(_tiny_awsim_osm())
    return str(path)


def test_load_awsim_map_local_coordinates(tiny_map):
    from torchdrivesim.lanelet2 import load_lanelet_map
    m = load_lanelet_map(tiny_map, origin=(0.0, 0.0), robust=True,
                         use_local_coordinates=True, recenter=True)
    assert len(list(m.laneletLayer)) == 2
    xs = [p.x for p in m.pointLayer]
    assert max(xs) - min(xs) == pytest.approx(60.0, abs=1.0)  # matches local_x span


def test_base_env_smoke(tiny_map):
    from awsim_rl_env import AWSIMDrivingEnv
    env = AWSIMDrivingEnv(tiny_map, num_agents=4, max_steps=10, seed=0)
    obs = env.reset()
    assert obs.shape == (4, env.OBS_DIM)
    obs2, reward, done, info = env.step(torch.zeros(4, env.ACT_DIM))
    assert obs2.shape == (4, env.OBS_DIM)
    assert reward.shape == (4,) and torch.isfinite(reward).all()
    assert done.shape == (4,)
    assert "active" in info and info["active"].shape == (4,)


def test_hetero_env_smoke(tiny_map):
    from awsim_hetero_rl_env import AWSIMHeteroDrivingEnv, TYPES
    env = AWSIMHeteroDrivingEnv(tiny_map, num_agents=4, max_steps=10, seed=0)
    obs = env.reset()
    assert obs.shape == (4, env.OBS_DIM)
    # Spawn quality (collision-free, on-surface) is exact on real maps but only
    # approximate on this degenerate 3.5 m x 60 m two-lanelet fixture, which has
    # too few spawn slots; assert it stays mostly clean rather than exactly zero.
    assert float((env.simulator.compute_collision()[0] > 0).float().mean()) <= 0.5
    assert float((env.simulator.compute_offroad()[0] > 0).float().mean()) <= 0.5
    _, reward, _, info = env.step(torch.zeros(4, env.ACT_DIM))
    assert torch.isfinite(reward).all()
    for t in TYPES:
        assert f"reached_{t}" in info


def test_spawns_rerandomise_each_reset(tiny_map):
    from awsim_hetero_rl_env import AWSIMHeteroDrivingEnv
    env = AWSIMHeteroDrivingEnv(tiny_map, num_agents=6, max_steps=10, seed=1)
    env.reset(); a = env._init_state.clone()
    env.reset(); b = env._init_state.clone()
    assert not torch.equal(a, b)


def test_goal_reached_freezes_agent(tiny_map):
    from awsim_rl_env import AWSIMDrivingEnv
    env = AWSIMDrivingEnv(tiny_map, num_agents=4, max_steps=20, seed=0, goal_radius=5.0)
    env.reset()
    st = env._state().clone()
    st[0, :2] = env.goals[0]            # place agent 0 on its goal
    env.simulator.set_state(st.unsqueeze(0))
    env._prev_dist = env._dist_to_goal(env._state())
    _, _, _, info = env.step(torch.ones(4, env.ACT_DIM))   # full throttle
    assert bool(env._reached[0]) and bool(info["active"][0])  # reaching step is active
    p0 = env._state()[0, :2].clone()
    for _ in range(5):
        _, _, _, info = env.step(torch.ones(4, env.ACT_DIM))
    assert float(torch.linalg.norm(env._state()[0, :2] - p0)) < 1e-3   # frozen
    assert not bool(info["active"][0])                                # excluded from training


@pytest.mark.parametrize("hetero", [False, True])
def test_train_smoke(tiny_map, tmp_path, hetero):
    from omegaconf import OmegaConf
    from awsim_rl_train import PPOConfig, train
    cfg = OmegaConf.structured(PPOConfig(
        map_path=tiny_map, save_dir=str(tmp_path / "out"), hetero=hetero,
        num_agents=4, updates=2, rollout_steps=16, max_steps=10, minibatches=2,
    ))
    train(cfg)  # should run end to end and write a policy
    assert os.path.exists(os.path.join(cfg.save_dir, "policy.pt"))
