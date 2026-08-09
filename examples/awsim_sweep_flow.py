"""Run several AWSIM training configurations concurrently and collect their results.

One GPU, so the runs share it: three at once is roughly the same total wall clock
as three in sequence, but every result lands together and the comparison can be
read in one go rather than three hours apart.

Each plan changes exactly one thing from the reference run (`hit-lane-300u`), so
whatever moves is attributable. Run it with a Prefect environment that is *not*
the project's own - `uv run` resyncs the project venv and would drop prefect:

    <prefect-venv>/bin/python examples/awsim_sweep_flow.py

The training itself is launched through `uv run` so it gets the project env.
"""
import argparse
import json
import os
import re
import subprocess
import time

from prefect import flow, get_run_logger, task
from prefect.concurrency.sync import concurrency
from prefect.task_runners import ThreadPoolTaskRunner

# One GPU. The limit is a *global* one so it holds across flow runs: submit a fourth
# plan while three are training and Prefect queues it rather than thrashing the card.
#   prefect gcl create gpu-train --limit 3
GPU_SLOT = "gpu-train"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = "nishishinjuku_autoware_map/lanelet2_map.osm"

# The reference: every plan below is this, with one field changed.
BASE = dict(
    hetero="true", device="cuda", map_path=MAP,
    num_agents=512, updates=300, rollout_steps=256, max_steps=250, n_parked=16,
    lr="5e-4", ent_coef=0.002, vf_coef=0.5, penalty_warmup=150,
    w_progress=0.1, w_goal=5.0, w_offroad=0.6, w_collision=6.0,
    w_redlight=0.5, w_wrongway=0.2, w_speeding=0.05, w_yield=0.4,
    w_proximity=0.1, ttc_threshold=3.0, w_lane=0.4,
    w_collision_level=0.0, lat_accel_max=4.0,
    w_lanechange=0.0, w_solidcross=0.0, max_steer_scale=1.0, goal_dist_scale=1.0,
    w_follow=0.0, terminate_on_teleport="false", offroad_fix="false",
    seed=42,
    value_norm="false", penalty_scale=1.0,
    checkpoint_every=50,
)

PLANS = {
    # Charging only the rise made staying overlapped free: contacts fell 14% but the
    # time spent overlapped rose 18%. A small level term restores a reason to get out.
    "escape": dict(w_collision_level=0.3),
    # The speed loss (vehicle speed/limit 0.73 -> 0.29) survived the lane-penalty fix,
    # so the steering cap is the remaining suspect. 6.0 m/s^2 allows a 9.4 m junction
    # radius at 7.5 m/s instead of 6.1.
    "steer": dict(lat_accel_max=6.0),
    # Rear-end is 53% of contacts at a 22 km/h closing speed while the anticipatory RSS
    # term costs ~0.013 per step. The 0.1-vs-0.3 A/B that chose 0.1 predates the
    # same-lane bend fix, the lane penalty and the lookahead.
    "prox": dict(w_proximity=0.3),
    # Changing lane was free: `offroad` needs the drivable surface, `wrongway` needs 90
    # degrees, and `_lane_offset` measures to the nearest lane of any kind, so the offset
    # resets the moment the agent is over the line. Stepping sideways was the cheapest
    # escape from a rear-end conflict, and crossing conflicts went 25.5% -> 39.7% of
    # contacts once driving into one got expensive.
    "lanechg": dict(w_lanechange=0.5, w_solidcross=3.0),
    # ...and the same, charging only what the map says is not allowed, to separate
    # "lane changes are being abused" from "illegal lane changes are being abused".
    "solidonly": dict(w_lanechange=0.0, w_solidcross=3.0),
    # The reference again, seed only. Five single-variable plans have now landed within
    # 0.089-0.097 of each other on contact rate and there is no replicate to say whether
    # that band is a result or the noise floor. This run is the ruler.
    "repeat": dict(seed=123),
    # Two untested causes of the speed loss (goals 3.65 -> 2.47, vehicle speed/limit
    # 0.73 -> 0.29). `steer` cleared the lateral-acceleration cap; these are what is
    # left: the lane penalty itself, and the geometric steering limit, which took the
    # car's minimum turn radius from 1.96 m to 6.6 m and was never varied.
    "nolane": dict(w_lane=0.0),
    "widesteer": dict(max_steer_scale=2.0),
    # `repeat` put the seed-only spread at 0.006 on contact rate and 0.08 on goals, which
    # is the size of every "effect" the six single-variable plans produced. Nothing since
    # hit300 has been distinguishable from noise. So: change something big enough to
    # matter, and run it twice.
    "balance": dict(penalty_scale=0.5),
    "balance2": dict(penalty_scale=0.5, seed=123),
    # MAPPO's most influential factor, and the value loss here sits at 2-3 without
    # falling while the return is assembled from thirteen terms of very different scale.
    "vnorm": dict(value_norm="true"),
    # Halving every penalty worked - goals 2.40-2.48 -> 2.72-2.74 and speed/limit 0.52 ->
    # 0.61 across two seeds, with contact unchanged - but it also halved the offroad
    # penalty, and offroad doubled while `lost` went 37-40 -> 73-98. The effect that
    # mattered was making driving positive-expected-value, so buy the same surplus from
    # the other side: leave the penalties alone and pay more for arriving.
    "boost": dict(w_goal=12.0),
    "boost2": dict(w_goal=12.0, seed=123),
    # vnorm's goals (2.57) and speed (0.56) both sit just above the two baselines but
    # inside a seed spread of 0.08. One replicate settles it.
    "vnorm2": dict(value_norm="true", seed=123),
    # arXiv:2606.19370 gets human-compatible driving out of self-play with +1 for the
    # goal, -1 for a collision or going off-road, and nothing else - "deliberately
    # avoiding dense shaping". Thirteen shaped terms is the opposite bet, and the
    # measurement that started this week says the shaped terms sum to more than the
    # incentive to drive. This is the other end of the axis `balance` moved along.
    "sparse": dict(w_progress=0.0, w_goal=1.0, w_collision=0.0, w_collision_level=1.0,
                   w_offroad=1.0, w_redlight=0.0, w_wrongway=0.0, w_speeding=0.0,
                   w_yield=0.0, w_proximity=0.0, w_lane=0.0, w_lanechange=0.0,
                   w_solidcross=0.0),
    # The goal sat 150 m away while gamma=0.98 gives a 5 s horizon - about 35 m at the
    # speed these agents drive - so the +5 for arriving was worth 0.074 discounted,
    # less than one step of the 0.111 penalty budget. The value function could not see
    # it. Worse, the bearing to a point that far along a curving route is a chord, not
    # the road: 34% of vehicle steps had the goal more than 30 degrees off the lane
    # direction, median 17.6 m of lateral offset against a 3 m lane. Shortening it to
    # 50 m takes the median offset to 2.1 m.
    "shortgoal": dict(goal_dist_scale=0.3333),
    "shortgoal2": dict(goal_dist_scale=0.3333, seed=123),
    # The ego block is now normalised by each agent's own goal spacing instead of a
    # fixed 100 m / 50 m, which alone took the clamp saturation from 62% to 5%. That
    # moves the baseline, so it needs re-measuring before anything is read against it.
    "base": dict(),
    # The corridor. Instead of a bearing to one point 150 m along one lane, the policy
    # is shown where each lane it may legally be in goes next - its own and the two it
    # could change into - and is paid for ground covered while tracking *any* of them.
    # The max over branches is the point: every lane is worth the same, so when to
    # change lane stays a decision rather than something the route dictates. Cars and
    # motorbikes only; a cyclist has 80 of 282 lanelets with a lane-changeable
    # neighbour and a pedestrian has none, so both keep the goal-distance term.
    # 0.2 puts the forward payment at about 0.098 per step at the speeds these agents
    # drive, against the 0.047 the progress term was paying and the 0.111 of penalties.
    # A respawn drops the agent somewhere unrelated, and `done` was never set for it, so
    # GAE bootstrapped V(that unrelated place) into the very step that earned the goal.
    # 1040 transitions an episode, and they are the ones that matter most. Terminating
    # them makes the last goal a real boundary for that agent, after which it starts
    # again from a freshly drawn spawn point.
    # `_offroad` is metres of overhang, not an indicator, and `progress * (1 - offroad)`
    # reversed the sign of the progress reward on 72% of off-road steps - median
    # multiplier -1.07, mean -9.5. Off the road, an agent was paid to stop and paid to
    # stay. Withhold the credit instead of reversing it, and cap the penalty, which was
    # running at 1.24 a step against a forward incentive of 0.095.
    "offroadfix": dict(offroad_fix="true"),
    "offroadfix2": dict(offroad_fix="true", seed=123),
    "teledone": dict(terminate_on_teleport="true"),
    "teledone2": dict(terminate_on_teleport="true", seed=123),
    # The corridor again, with the off-road withholding that its reward path never had.
    # `w_follow` replaces `progress` for wheeled agents and nothing gated it on being on
    # the road, so a car off the map still collected the follow reward in full - which
    # is the obvious candidate for why `lost` tripled, 24 -> 66/86, in the first attempt.
    # Read against `offroadfix`, not `base`, so the corridor is the only difference.
    "corridorfix": dict(w_follow=0.2, offroad_fix="true"),
    "corridorfix2": dict(w_follow=0.2, offroad_fix="true", seed=123),
    "corridor": dict(w_follow=0.2),
    "corridor2": dict(w_follow=0.2, seed=123),
    # "Always use observation normalization" (arXiv:2006.05990). Scale only, no
    # centring: the partner and road blocks are zero-padded and the deep-sets max-pool
    # reads an exact zero as "nobody there", which centring would destroy. Expect
    # little - measured over 512 agents the per-feature RMS already spans only
    # 0.038-1.06, so the hand-picked scalings have done most of this job already.
    "obsnorm": dict(obs_norm="true"),
    "obsnorm2": dict(obs_norm="true", seed=123),
    # The other half of the same measurement. gamma=0.98 with dt=0.1 is a 5 s horizon,
    # about 35 m at the speed these agents drive, against a goal 150 m away - the +5 for
    # arriving discounts to 0.074, less than one step of penalties. `shortgoal` brings
    # the goal inside the horizon; this stretches the horizon to the goal instead:
    # 1/(1-0.995) = 200 steps = 140 m. Left alone, `goal_dist` stays at 150 m.
    # Watch the value loss - the return grows about fourfold in magnitude and nothing
    # normalises it, which is the case `vnorm` was meant to cover.
    "gamma995": dict(gamma=0.995),
    "gamma995b": dict(gamma=0.995, seed=123),
    "sparse2": dict(w_progress=0.0, w_goal=1.0, w_collision=0.0, w_collision_level=1.0,
                    w_offroad=1.0, w_redlight=0.0, w_wrongway=0.0, w_speeding=0.0,
                    w_yield=0.0, w_proximity=0.0, w_lane=0.0, w_lanechange=0.0,
                    w_solidcross=0.0, seed=123),
}

DONE = re.compile(r"^\[done\] per-type (goals|v/limit)/agent?: (.*)$")
UPD = re.compile(r"^upd +(\d+)/(\d+) \| return +(\S+) \| goals +(\S+).*?coll (\S+).*?"
                 r"off (\S+).*?lane (\S+) hit (\S+)")


@task(retries=0, log_prints=True)
def train(name: str, overrides: dict, tag: str) -> dict:
    """One training run. Returns the tail metrics so the flow can compare them."""
    logger = get_run_logger()
    cfg = dict(BASE, **overrides)
    save_dir = f"./awsim_rl_output/{tag}-{name}"
    cfg.update(save_dir=save_dir, wandb="true", wandb_project="awsim-rl",
               wandb_run=f"{tag}-{name}")
    args = [f"{k}={v}" for k, v in cfg.items()]
    log_path = os.path.join(REPO, f"awsim_rl_output/{tag}-{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
    logger.info("start %s: %s", name, " ".join(f"{k}={overrides[k]}" for k in overrides))
    t0 = time.time()
    with concurrency(GPU_SLOT, occupy=1):
        logger.info("%s got a GPU slot", name)
        with open(log_path, "w") as log:
            rc = subprocess.call(["uv", "run", "python", "examples/awsim_rl_train.py"] + args,
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
    mins = (time.time() - t0) / 60
    tail, per_type = [], {}
    for line in open(log_path, errors="ignore"):
        m = UPD.match(line)
        if m:
            tail.append(dict(zip(("upd", "total", "return", "goals", "coll", "off",
                                  "lane", "hit"), m.groups())))
        d = DONE.match(line)
        if d:
            per_type[d.group(1)] = d.group(2)
    last = tail[-20:] or [{}]
    mean = {k: sum(float(t[k]) for t in last) / len(last)
            for k in ("return", "goals", "coll", "off", "lane", "hit") if last[0]}
    result = dict(name=name, overrides=overrides, rc=rc, minutes=round(mins, 1),
                  updates=len(tail), tail20=mean, per_type=per_type, log=log_path)
    logger.info("done %s in %.0f min: %s", name, mins, json.dumps(mean))
    return result


@flow(name="awsim-plan-sweep", task_runner=ThreadPoolTaskRunner(max_workers=3))
def sweep(plans: str = "", tag: str = "plan"):
    """Submit every plan at once; they share the GPU and finish together."""
    logger = get_run_logger()
    chosen = [p for p in (plans.split(",") if plans else PLANS) if p in PLANS]
    logger.info("running %d plans on one GPU: %s", len(chosen), ", ".join(chosen))
    futures = [train.submit(name, PLANS[name], tag) for name in chosen]
    results = [f.result() for f in futures]
    out = os.path.join(REPO, f"awsim_rl_output/{tag}-summary.json")
    json.dump(results, open(out, "w"), indent=1)
    for r in sorted(results, key=lambda r: -float(r["tail20"].get("goals", 0))):
        logger.info("%-8s %s | %s", r["name"], json.dumps(r["overrides"]),
                    json.dumps(r["tail20"]))
    logger.info("summary -> %s", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="", help="comma-separated subset of PLANS")
    ap.add_argument("--tag", default="plan")
    a = ap.parse_args()
    sweep(plans=a.plans, tag=a.tag)
