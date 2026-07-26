# AWSIM × TorchDriveSim RL — Handover Notes

Handover for continuing this work on a **GPU** machine. It records what was built,
why, the non-obvious findings, current results, and the recommended next steps.
Everything here lives on branch `claude/awsim-ll2-traffic-simulation-3de141`.

## TL;DR

We took an **AWSIM / Autoware Lanelet2 map** (the Quick Start "Nishi-Shinjuku" map),
ingested it into **TorchDriveSim**, built a traffic simulation (mp4 output), and then a
**heterogeneous multi-agent RL** setup (vehicles / motorcycles / cyclists / pedestrians
trained by one shared, type-conditioned policy) following the **GPUDrive / Nocturne /
PufferDrive** recipe. Everything runs and learns on CPU in smoke/short runs; it is now
ready for **real GPU training**, which was intentionally left to the GPU operator.

## Environment setup

Python 3.11. Install the CUDA-matched `torch` first (see project README), then:

```bash
pip install -r examples/requirements-awsim.txt   # lanelet2, opencv, imageio-ffmpeg, ...
```

- `lanelet2` (PyPI 1.2.3) provides the Python bindings — no ROS needed.
- `imageio-ffmpeg` is required for mp4 output; `matplotlib` only for the reward curve.
- The RL env/PPO themselves need only `torch` + `torchdrivesim` + `lanelet2`.

Run examples with the repo root and `examples/` on `PYTHONPATH` (the example files
import helpers from each other):

```bash
export PYTHONPATH=/path/to/torchdrivesim:/path/to/torchdrivesim/examples:$PYTHONPATH
```

### Getting the AWSIM Nishi-Shinjuku map

The actual Quick Start map is a GitHub release asset (works over plain HTTPS):

```bash
curl -L -o nishishinjuku_autoware_map.zip \
  https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip
unzip nishishinjuku_autoware_map.zip
# -> nishishinjuku_autoware_map/lanelet2_map.osm   (~10 MB, 36,936 nodes, ~1.1 x 1.1 km)
```

Small quick-to-render alternative (Autoware validator sample, same VMB dialect,
~140 x 89 m) — used for fast CPU smoke tests here:

```bash
curl -L -o sample_map.osm \
  https://raw.githubusercontent.com/tier4/autoware_lanelet2_map_validator/main/autoware_lanelet2_map_validator/test/data/map/sample_map.osm
```

## Files (all under `examples/`, plus one library change)

| File | Role |
|------|------|
| `awsim_lanelet2_traffic.py` | Map loading + mesh + **scripted** lane-following traffic sim, mp4/gif output. Also the shared helper module (`map_latlon_origin`, `build_driving_surface_mesh`, `build_route`, `mesh_camera`, `polyline_cumlen`, `point_at_arclen`, `save_video`, `_attr`) imported by the others. |
| `awsim_rl_env.py` | `AWSIMDrivingEnv` — base single-type RL env (all vehicles). Observation = ego + neighbours + road graph; reward = goal + progress − collision − offroad; goal termination + freeze. |
| `awsim_hetero_rl_env.py` | `AWSIMHeteroDrivingEnv(AWSIMDrivingEnv)` — adds 4 road-user types (compound kinematics), type-aware spawning, per-type observation/stats. |
| `awsim_rl_train.py` | Compact PPO (`ActorCritic` with ego/partner/road encoders + per-type heads) + training loop + smoke test + reward curve + rollout video. |
| `torchdrivesim/lanelet2.py` | `load_lanelet_map(...)` extended (see below). **Only library change; backward-compatible.** |

## Key finding #1 — ingesting AWSIM/Autoware maps

AWSIM maps are exported by **Vector Map Builder** (`generator="VMB"`) and differ from the
bundled CARLA maps in two ways that break TorchDriveSim's stock loader:

1. **Projection.** They are geo-referenced (real lat/lon around a Japanese origin).
   TorchDriveSim's default `UtmProjector(Origin(0,0))` lands in UTM zone 31 and rejects
   longitudes ~140°E outright. Fix: project from a point inside the map (mean node lat/lon).
   The nodes also carry `local_x`/`local_y` tags = the MGRS-projected metres Autoware
   actually uses; reading those back is numerically identical to Autoware's `MGRSProjector`.
2. **Regulatory elements.** Autoware-specific reg-elems (`detection_area`,
   `no_stopping_area`, `crosswalk`, `virtual_traffic_light`, …) are not implemented by
   upstream lanelet2, so a strict `load` throws. Fix: `loadRobust` skips them; geometry
   still loads.

`load_lanelet_map` now supports (all default-off, so CARLA behavior is unchanged):

```python
load_lanelet_map(path, origin=(lat, lon), robust=True,
                 use_local_coordinates=True,   # honour local_x/local_y (Autoware MGRS)
                 recenter=True,                 # subtract centroid (large MGRS offsets)
                 projector=None)                # pass Autoware MGRSProjector if available
```

Autoware uses right-handed ENU, so — unlike CARLA — the map is **not** inverted and lane
markings are built with `left_handed=False`.

`lanelet2_extension_python` (the real MGRS projector) is **not on PyPI**; the
`local_x`/`local_y` path is the equivalent used here.

## Key finding #2 — RL design (GPUDrive / Nocturne / PufferDrive port)

Reference: `github.com/Emerge-Lab/Adaptive_Driving_Agent` (PufferDrive/GPUDrive). We read
its C/torch source and ported the design to TorchDriveSim. PPO hyper-params come from its
`drive.ini`: `gamma=0.98, gae_lambda=0.95, clip=0.2, ent_coef=0.005, vf_coef=2.0, lr=3e-3`.

**Observation** (flat vector `[ego | partners | road]`, all agents = parallel streams):
- **ego** (`EGO_DIM`): base 8 = `[speed, dist_to_goal, head_err_cos, head_err_sin, goal_x_ego, goal_y_ego, prev_accel, prev_steer]`. Hetero appends own type one-hot (→ 12).
- **partners** (`MAX_PARTNERS=8 × PARTNER_FEATURES`): K nearest agents within `VIEW_RADIUS=30 m`, ego frame, zero-padded. Base 7 = `[rel_x, rel_y, width, length, rel_head_cos, rel_head_sin, signed_speed]`; hetero appends neighbour type one-hot (→ 11).
- **road** (`MAX_ROAD=10 × ROAD_FEATURES=4`): K nearest lane-centreline points within `ROAD_RADIUS=30 m`, ego frame = `[rel_x, rel_y, rel_dir_cos, rel_dir_sin]`.
- Dims: base `OBS_DIM=104`, hetero `OBS_DIM=140`. The env exposes `EGO_DIM, MAX_PARTNERS, PARTNER_FEATURES, MAX_ROAD, ROAD_FEATURES, NUM_TYPES, TYPE_ONEHOT_SLICE` for the policy.

**Policy** (`ActorCritic`): separate **ego / partner / road encoders**; partner & road sets
are **max-pooled** (permutation-invariant deep-sets) → trunk → **per-type actor heads**
(one mean+log_std per type; value head shared). Type is read from the ego one-hot in the obs.

**Reward** (per agent, per step): `progress (Δdistance-to-goal) − 0.5·collision − 0.5·offroad + 1.0·(reached goal)`.

**Goal termination** (important, was a real bug we fixed): on reaching `goal_radius`, the
agent gets +1 **once**, is marked done, and is **frozen** (action zeroed + speed set to 0
via `Simulator.set_state`) so it stops moving and stops interfering. `step()` returns an
`active` mask; the trainer trains PPO **only on pre-terminal transitions** (post-goal steps
excluded).

**Spawning** (`AWSIMHeteroDrivingEnv`): per type, using lanelet2 **participant traffic
rules** (`traffic_rules.canPass`, honours Autoware `participant:*` tags + subtypes) and that
participant's **routing graph**:
- vehicle → `Vehicle`, motorcycle → `VehicleMotorcycle` (fallback Vehicle), cyclist →
  `Bicycle`, pedestrian → `Pedestrian`.
- Headings follow lanelet direction (maps are mostly one_way → no wrong-way spawns).
- Start positions sampled **along** the lane, **re-randomised every reset**, and
  **rejection-sampled to be collision-free** (per-type footprint). Route pools + arc-length
  caches are precomputed once (`pool_cap`) so resets stay cheap.

**Agent types** (`TYPE_SPEC` in `awsim_hetero_rl_env.py`): vehicle/motorcycle/cyclist use
`KinematicBicycle`; pedestrian uses `SimpleKinematicModel` (omnidirectional). Mixed in one
scene via `CompoundKinematicModel` (action is the 4-dim superset; bicycle uses `[:2]`).
Per-type size, turn radius, top speed (clamped each step in `_post_physics`), colour.

## How to run

Traffic sim (scripted, mp4):
```bash
python examples/awsim_lanelet2_traffic.py \
  map_path=nishishinjuku_autoware_map/lanelet2_map.osm fov=340 video_format=mp4
```

RL smoke test (CPU, a few updates, saves rollout mp4 + reward curve):
```bash
python examples/awsim_rl_train.py map_path=sample_map.osm hetero=true smoke_test=true
```

Full GPU training (the next step):
```bash
python examples/awsim_rl_train.py hetero=true \
  map_path=nishishinjuku_autoware_map/lanelet2_map.osm \
  device=cuda num_agents=64 updates=3000 rollout_steps=256 \
  checkpoint_every=100          # writes policy_<n>.pt; resume=<path> to warm-start
```
Metrics can go to **Weights & Biases** (`pip install wandb`):
```bash
python examples/awsim_rl_train.py hetero=true device=cuda ... \
  wandb=true wandb_project=awsim-rl wandb_run=shinjuku-64a
```
It logs return, reached (overall + per type), collision, offroad and losses per update,
and the final rollout video + reward curve. Disabled by default; `wandb` is only imported
when `wandb=true`.

Config knobs (OmegaConf dot-list): `updates, rollout_steps, num_agents, max_steps, dt,
gamma, gae_lambda, clip_coef, ent_coef, vf_coef, lr, update_epochs, minibatches, hidden,
seed, hetero, smoke_test, checkpoint_every, resume, wandb, wandb_project, wandb_run`. Env knobs live in the env `__init__`
(`mix`, `spawn_gap`, `pool_cap`, `goal_radius`, `MAX_PARTNERS`/`MAX_ROAD`/radii as class
attrs). Progress prints are flushed, so `tail -f` works during long runs.

The code is **device-correct for CUDA** (all env tensors carry `device=`, index tensors are
on-device, GAE runs on the rollout device); verified on CPU, and audited for the CUDA path.
Building the env on the full Shinjuku map takes **~60-70 s once** (four per-participant
routing graphs + route pools + the road-point cloud); the per-step cost is small. On CPU
the training bottleneck is the rollout; a GPU makes the deep-sets policy and larger
`num_agents` practical.

## Tests

`tests/test_awsim_examples.py` — self-contained smoke tests (generate a tiny AWSIM-format
map, no network): map load with local coords, base + hetero env construct/reset/step and
obs dims, spawn re-randomisation, goal freeze + `active` mask, and a 2-update PPO run for
both single- and multi-type policies. Run:
```bash
python -m pytest tests/test_awsim_examples.py -q     # 7 passed (~7 s)
```

## Current results (CPU, short runs — NOT converged)

- Map ingestion verified on the real Shinjuku map: 979 drivable lanelets, extent
  1113 × 1086 m, geometry matches the file's raw `local_x/local_y` exactly.
- Scripted 24-agent traffic sim renders correctly (mp4).
- RL learns: on `sample_map`, mean episode return rises from about −9 to +5…+29 depending
  on feature set; goal-reaching 0 → 0.5–0.9 over 50–60 CPU updates.
- Heterogeneous, all features (neighbours + road + per-type heads), 50 CPU updates, 16
  agents: return −9 → ~+5 (still climbing), final greedy reached ≈ 0.56 (vehicle 1.0,
  pedestrian 0.25, moto/cyclist 0 — only 3 each, under-trained).

## Known limitations / open issues (good first tasks on GPU)

1. **Under-trained on CPU.** The rich obs (104/140-dim) + 3 encoders + per-type heads need
   many more updates than we could run on CPU. First GPU job: reproduce a clean learning
   curve with `updates≈2000–3000`, `num_agents≈64`, and confirm minority types (moto,
   cyclist, pedestrian) actually improve with the per-type heads.
2. **Offroad still non-zero** at intersections — agents cut corners. Road-graph obs helps;
   consider adding a **lane-alignment reward** (PufferDrive `reward_lane_align`) or a
   heavier offroad weight. Tune per type.
3. **Pedestrians lag.** Different kinematics + short crosswalk goals. Per-type heads are in;
   next levers are per-type reward weights / goal radius, or a pedestrian-specific curriculum.
4. **Rendering is the training bottleneck for full-map viz**, not training itself (RL loop
   doesn't render except when saving video). Full Shinjuku lane-mesh build is ~30 s (once);
   per-frame render ~1–2 s on CPU → use a GPU renderer (pytorch3d/nvdiffrast) for videos, or
   render a cropped region.
5. **`reset()` rebuilds the Simulator each episode** (renderer + kinematic model). Left as-is
   for correctness/simplicity; if reset overhead matters at scale, reuse the simulator and
   only `set_state`/reset masks (mirrors `gym_env.py`'s copy pattern). Flagged, not done.
6. **Spawn realism TODO** (not yet done): strict one-way handling for `one_way:no` lanelets,
   longitudinal car-spacing / stop-line avoidance, per-lanelet `speed_limit` → per-agent
   vmax, pedestrian dwell on walkways.
7. **No road-graph in the base (vehicle-only) env's spawn variety** beyond routes — fine, but
   note base env spawns are fixed at init (only hetero re-randomises spawns each reset).

## Environment/tooling gotchas discovered

- This sandbox's egress **blocked** `autowarefoundation.github.io`, `codeload.github.com`,
  `patch-diff.githubusercontent.com`, `download.pytorch.org`, and the GitHub **API** was
  repo-scoped. **Open**: `raw.githubusercontent.com` (any repo), GitHub **release-asset**
  hosts, and PyPI. That's how the map was fetched. A GPU box likely has open egress — ignore
  if so.
- `lanelet2` Python: iterating `lanelet.attributes` yields `(key,value)` entries, not keys;
  use `k in l.attributes` / `l.attributes[k]`. `AttributeMap` has no `.get`.
- `lanelet2.geometry.interpolatedPointAtDistance` works on a single centerline, not on our
  chained numpy route polylines — hence the small `point_at_arclen` helper.
- `compute_collision()` ignores `present_mask` (computes for all agents), so "removing" an
  agent by masking doesn't drop it from collisions — we freeze reached agents instead.
- `Date.now`-style nondeterminism isn't relevant here; RNG is seeded via `np.random.default_rng`.

## Commit history (branch `claude/awsim-ll2-traffic-simulation-3de141`)

```
9123ed4 Add per-type policy heads for heterogeneous agents
e8e2f18 Add road-graph observation (GPUDrive-style lane-centreline points)
204bd43 Terminate and freeze agents at their goal; exclude post-goal steps from training
03fcde4 Add neighbour observations (GPUDrive/Nocturne-style) so agents see each other
6fc5c11 Simplify examples: dedup env classes, share helpers, small efficiency fixes
06a8b4b Use per-participant traffic rules and routing graphs for spawning
7429483 Type-aware, randomized, collision-free spawning for heterogeneous RL
8f84b15 Add heterogeneous multi-agent RL (vehicle/motorcycle/cyclist/pedestrian)
363226c Add PPO RL training scaffold on AWSIM maps, and mp4 video output
ad32b48 Point AWSIM example at the real Nishi-Shinjuku Quick Start map
eba9415 Support Autoware MGRSProjector in map loading, with equivalent fallback
bfbd03c Load AWSIM/Autoware maps via local_x/local_y (Autoware projector semantics)
3146ad3 Add AWSIM/Autoware Lanelet2 traffic simulation example
```

## Suggested first GPU session

1. `pip install` deps with a CUDA torch build; download the Shinjuku map.
2. Sanity: `python examples/awsim_rl_train.py map_path=sample_map.osm hetero=true smoke_test=true`.
3. Full run on Shinjuku (command above), `device=cuda`. Watch the printed per-type reached
   rates and the saved `reward_curve.png` / `awsim_rl_rollout.mp4` in `save_dir`.
4. If minority types stay flat, tune `mix`, per-type reward weights, and consider a
   lane-alignment reward. Then iterate on the spawn-realism TODOs.
