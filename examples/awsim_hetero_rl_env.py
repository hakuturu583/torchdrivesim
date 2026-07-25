"""
Heterogeneous multi-agent RL environment on an AWSIM / Autoware map.

Extends awsim_rl_env with several road-user *types* trained together by one
shared, type-conditioned policy - the recipe used by heterogeneous driving
simulators such as GPUDrive, Nocturne, Waymax, SMARTS and MetaDrive, which all
mix vehicles, cyclists and pedestrians in a single scene.

Four types, two kinematic families (via CompoundKinematicModel):
    vehicle, motorcycle, cyclist  -> KinematicBicycle  (action = accel, steer)
    pedestrian                    -> SimpleKinematicModel (omnidirectional)

Type-aware spawning (see `_build_spawn_pools` / `_sample_spawns`):
  * spawn surface is chosen per type from the lanelet subtypes / Autoware
    participant tags - vehicles & motorcycles on `road`, cyclists on
    `road_shoulder`/`road`, pedestrians on `crosswalk`/`walkway`
    (participant:pedestrian);
  * headings follow the lanelet direction (Autoware maps are mostly one_way),
    so wheeled agents never spawn against traffic;
  * start positions are sampled *along* the lane (not always the start) and are
    re-randomised every reset for RL generalisation;
  * spawns are rejection-sampled to be collision-free, with a per-type footprint.

A one-hot type code is appended to each observation. Reward is shared:
    goal(+1) + progress - collision(0.5) - offroad(0.5).
"""
import numpy as np
import torch

from torchdrivesim.kinematic import KinematicBicycle, SimpleKinematicModel, CompoundKinematicModel
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.utils import Resolution
from torchdrivesim.lanelet2 import load_lanelet_map
import lanelet2
from lanelet2.traffic_rules import Locations, Participants

from awsim_lanelet2_traffic import (
    map_latlon_origin, build_driving_surface_mesh, build_route, _attr,
)

# type -> kinematic family, size, turn radius, top speed, lanelet2 participant,
# goal distance, colour. The participant selects which lanelets the type may
# spawn/route on (via traffic_rules.canPass) and which routing graph is used,
# so Autoware participant tags + subtypes are honoured directly.
TYPES = ["vehicle", "motorcycle", "cyclist", "pedestrian"]
TYPE_SPEC = {
    "vehicle":    dict(model=0, size=(4.97, 2.04), lr=1.96, vmax=14.0, participant="vehicle",    goal_dist=45.0, color=(32, 74, 135)),
    "motorcycle": dict(model=0, size=(2.20, 0.90), lr=0.80, vmax=18.0, participant="motorcycle", goal_dist=45.0, color=(230, 90, 20)),
    "cyclist":    dict(model=0, size=(1.80, 0.70), lr=0.60, vmax=6.0,  participant="bicycle",     goal_dist=30.0, color=(24, 104, 225)),
    "pedestrian": dict(model=1, size=(0.70, 0.70), lr=0.50, vmax=2.0,  participant="pedestrian",  goal_dist=None, color=(173, 127, 168)),
}
DEFAULT_MIX = {"vehicle": 0.4, "motorcycle": 0.15, "cyclist": 0.15, "pedestrian": 0.3}
# lanelet2 Participants used per participant key (with graceful fallback to Vehicle)
PARTICIPANT_ENUM = {
    "vehicle": "Vehicle", "motorcycle": "VehicleMotorcycle",
    "bicycle": "Bicycle", "pedestrian": "Pedestrian",
}


def _polyline_length(poly):
    seg = np.diff(poly, axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _arc_interp(poly, s):
    """Point (x, y, heading) at arc-length s along a polyline."""
    seg = np.diff(poly, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = max(0, min(int(np.searchsorted(cum, s) - 1), len(seg) - 1))
    r = (s - cum[i]) / max(seglen[i], 1e-6)
    x, y = poly[i] + r * seg[i]
    return x, y, float(np.arctan2(seg[i, 1], seg[i, 0]))


class AWSIMHeteroDrivingEnv:
    OBS_DIM = 8 + len(TYPES)  # base features + one-hot type
    ACT_DIM = 4               # superset action (bicycle uses [:2], pedestrian [:4])

    def __init__(self, map_path, num_agents=12, max_steps=80, dt=0.1, device='cpu',
                 goal_radius=3.0, mix=None, render_fov=None, render_res=512, seed=0,
                 spawn_attempts=25, spawn_gap=1.5, pool_cap=120):
        self.device = device
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.dt = dt
        self.goal_radius = goal_radius
        self.render_res = render_res
        self.spawn_attempts = spawn_attempts
        self.spawn_gap = spawn_gap
        self.pool_cap = pool_cap
        self._rng = np.random.default_rng(seed)
        mix = mix or DEFAULT_MIX

        origin = map_latlon_origin(map_path)
        self.lanelet_map = load_lanelet_map(map_path, origin=origin, robust=True,
                                            use_local_coordinates=True, recenter=True)
        self.mesh = build_driving_surface_mesh(self.lanelet_map).to(device)
        vx, vy = self.mesh.verts[..., 0], self.mesh.verts[..., 1]
        self._center = (float((vx.min() + vx.max()) / 2), float((vy.min() + vy.max()) / 2))
        self.render_fov = render_fov if render_fov is not None else \
            1.1 * max(float(vx.max() - vx.min()), float(vy.max() - vy.min()))

        # assign a fixed type to each agent (types don't change across episodes)
        counts = self._allocate(num_agents, mix)
        self.agent_type_idx = np.concatenate([[i] * counts[t] for i, t in enumerate(TYPES)]).astype(int)
        self._rng.shuffle(self.agent_type_idx)

        # static per-agent properties derived from type
        self.model_assignments = torch.tensor(
            [[TYPE_SPEC[TYPES[t]]["model"] for t in self.agent_type_idx]], device=device)
        self.vmax = torch.tensor([TYPE_SPEC[TYPES[t]]["vmax"] for t in self.agent_type_idx],
                                 dtype=torch.float32, device=device)
        self.lr_all = torch.tensor([TYPE_SPEC[TYPES[t]]["lr"] for t in self.agent_type_idx],
                                   dtype=torch.float32, device=device)
        sizes = np.stack([TYPE_SPEC[TYPES[t]]["size"] for t in self.agent_type_idx])
        self.agent_size = torch.tensor(sizes, dtype=torch.float32, device=device).unsqueeze(0)
        self.type_onehot = torch.zeros(num_agents, len(TYPES), device=device)
        self.type_onehot[torch.arange(num_agents), torch.tensor(self.agent_type_idx)] = 1.0

        self._build_spawn_pools()
        self.reset()

    # ------------------------------------------------------ spawn machinery
    def _allocate(self, n, mix):
        counts = {t: int(np.floor(n * mix.get(t, 0))) for t in TYPES}
        while sum(counts.values()) < n:
            counts["vehicle"] += 1
        return counts

    def _traffic_rules(self, participant_key):
        """traffic_rules for a participant key, falling back to Vehicle if unsupported."""
        name = PARTICIPANT_ENUM.get(participant_key, "Vehicle")
        for candidate in (name, "Vehicle"):
            try:
                return lanelet2.traffic_rules.create(Locations.Germany, getattr(Participants, candidate))
            except Exception:
                continue
        return lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)

    def _passable_lanelets(self, rules):
        lls = sorted([l for l in self.lanelet_map.laneletLayer if rules.canPass(l)], key=lambda l: l.id)
        if len(lls) > self.pool_cap:
            idx = sorted(self._rng.choice(len(lls), self.pool_cap, replace=False))
            lls = [lls[i] for i in idx]
        return lls

    def _build_spawn_pools(self):
        """One route-polyline pool per participant, built once from that
        participant's traffic rules (canPass) and routing graph."""
        participants = {TYPE_SPEC[t]["participant"] for t in TYPES}
        self._pools = {}
        for pkey in participants:
            rules = self._traffic_rules(pkey)
            graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
            polys = []
            for ll in self._passable_lanelets(rules):
                if pkey == "pedestrian":  # pedestrians cross a single lanelet
                    poly = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
                else:                      # wheeled agents follow a legal downstream route
                    poly = build_route(graph, ll)
                if poly.shape[0] >= 2 and _polyline_length(poly) > 3.0:
                    polys.append(poly)
            self._pools[pkey] = polys
        for pkey in participants:  # any empty pool falls back to the vehicle network
            if not self._pools[pkey]:
                self._pools[pkey] = self._pools.get("vehicle", [])

    def _sample_spawns(self):
        placed = []  # (x, y, radius)
        starts, headings, goals = [], [], []
        for tidx in self.agent_type_idx:
            spec = TYPE_SPEC[TYPES[tidx]]
            pool = self._pools[spec["participant"]] or self._pools["vehicle"]
            L, W = spec["size"]
            radius = 0.5 * max(L, W) + self.spawn_gap
            chosen = None
            for _ in range(self.spawn_attempts):
                poly = pool[self._rng.integers(len(pool))]
                total = _polyline_length(poly)
                if spec["goal_dist"] is None:            # pedestrian: cross whole lanelet
                    p = poly[::-1] if self._rng.random() < 0.5 else poly
                    s0, sg = 0.0, _polyline_length(p)
                else:                                     # wheeled: start along lane, goal ahead
                    p = poly
                    s0 = self._rng.uniform(0.0, max(total - 3.0, 0.0) * 0.6)
                    sg = min(s0 + spec["goal_dist"], total)
                x, y, h = _arc_interp(p, s0)
                gx, gy, _ = _arc_interp(p, sg)
                if all((x - px) ** 2 + (y - py) ** 2 > (radius + pr) ** 2 for px, py, pr in placed):
                    chosen = (x, y, h, gx, gy)
                    break
            if chosen is None:  # accept the last attempt if all overlapped
                chosen = (x, y, h, gx, gy)
            placed.append((chosen[0], chosen[1], radius))
            starts.append(chosen[:2]); headings.append(chosen[2]); goals.append(chosen[3:5])

        A = self.num_agents
        self.goals = torch.tensor(np.stack(goals), dtype=torch.float32, device=self.device)
        self._init_state = torch.zeros(1, A, 4, device=self.device)
        self._init_state[0, :, :2] = torch.tensor(np.stack(starts), dtype=torch.float32)
        self._init_state[0, :, 2] = torch.tensor(np.stack(headings), dtype=torch.float32)

    def _build_simulator(self):
        A = self.num_agents
        assign = self.model_assignments
        wheeled, ped = (assign[0] == 0), (assign[0] == 1)
        bike = KinematicBicycle(dt=self.dt)
        bike.set_params(lr=self.lr_all[wheeled].clone())
        bike.set_state(self._init_state[0, wheeled].clone())
        walk = SimpleKinematicModel(dt=self.dt, max_dx=TYPE_SPEC["pedestrian"]["vmax"])
        walk.set_state(self._init_state[0, ped].clone())
        kin = CompoundKinematicModel([bike, walk], model_assignments=assign, dt=self.dt)

        renderer = renderer_from_config(RendererConfig(left_handed_coordinates=False))
        for t in TYPES:
            renderer.color_map[t] = TYPE_SPEC[t]["color"]
            renderer.rendering_levels.setdefault(t, 4)
        cfg = TorchDriveConfig(left_handed_coordinates=False,
                               renderer=RendererConfig(left_handed_coordinates=False))
        self.simulator = Simulator(
            cfg=cfg, road_mesh=self.mesh, kinematic_model=kin, agent_size=self.agent_size,
            initial_present_mask=torch.ones(1, A, dtype=torch.bool, device=self.device),
            renderer=renderer, lanelet_map=[self.lanelet_map],
            agent_types=torch.tensor(self.agent_type_idx, device=self.device).view(1, A),
            agent_type_names=TYPES,
        )

    # ------------------------------------------------------------------- core
    def _state(self):
        return self.simulator.get_state()[0]

    def _dist_to_goal(self, state):
        return torch.linalg.norm(state[:, :2] - self.goals, dim=-1)

    def _observation(self, state, prev_action):
        x, y, psi, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        dx, dy = self.goals[:, 0] - x, self.goals[:, 1] - y
        c, s = torch.cos(psi), torch.sin(psi)
        gx_e, gy_e = c * dx + s * dy, -s * dx + c * dy
        dist = torch.linalg.norm(torch.stack([dx, dy], -1), dim=-1)
        head_err = torch.atan2(gy_e, gx_e)
        base = torch.stack([
            v / 10.0, dist.clamp(max=100.0) / 50.0,
            torch.cos(head_err), torch.sin(head_err),
            (gx_e / 50.0).clamp(-2, 2), (gy_e / 50.0).clamp(-2, 2),
            prev_action[:, 0], prev_action[:, 1],
        ], dim=-1)
        return torch.cat([base, self.type_onehot], dim=-1)

    def reset(self):
        self._sample_spawns()          # re-randomise spawns every episode
        self._build_simulator()
        self._t = 0
        self._prev_action = torch.zeros(self.num_agents, self.ACT_DIM, device=self.device)
        self._reached = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        state = self._state()
        self._prev_dist = self._dist_to_goal(state)
        return self._observation(state, self._prev_action)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(-1, 1)
        self.simulator.step(action.unsqueeze(0))
        state = self._state()
        capped_v = torch.clamp(state[:, 3], -self.vmax, self.vmax)  # per-type top speed
        if not torch.equal(capped_v, state[:, 3]):
            new_state = state.clone(); new_state[:, 3] = capped_v
            self.simulator.kinematic_model.set_state(new_state.unsqueeze(0))
            state = self._state()

        self._t += 1
        dist = self._dist_to_goal(state)
        progress = (self._prev_dist - dist)
        collision = (self.simulator.compute_collision()[0] > 0).float()
        offroad = (self.simulator.compute_offroad()[0] > 0).float()
        newly = (dist < self.goal_radius) & (~self._reached)
        reward = progress - 0.5 * collision - 0.5 * offroad + 1.0 * newly.float()
        reward = torch.where(self._reached, torch.zeros_like(reward), reward)
        self._reached |= (dist < self.goal_radius)
        self._prev_dist, self._prev_action = dist, action
        done = self._reached.clone() | (self._t >= self.max_steps)

        ti = torch.tensor(self.agent_type_idx, device=self.device)
        info = {'reached': float(self._reached.float().mean()),
                'collision': float(collision.mean()), 'offroad': float(offroad.mean())}
        for i, t in enumerate(TYPES):
            m = (ti == i)
            info[f'reached_{t}'] = float(self._reached[m].float().mean()) if m.any() else 0.0
        return self._observation(state, action), reward, done, info

    def render_frame(self):
        cam = torch.tensor([[list(self._center)]], device=self.device)
        psi = torch.zeros(1, 1, 1, device=self.device)
        img = self.simulator.render(camera_xy=cam, camera_psi=psi,
                                    res=Resolution(self.render_res, self.render_res), fov=self.render_fov)
        return img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
