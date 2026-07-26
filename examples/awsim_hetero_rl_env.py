"""
Heterogeneous multi-agent RL environment on an AWSIM / Autoware map.

Subclasses ``AWSIMDrivingEnv`` and adds several road-user *types* trained
together by one shared, type-conditioned policy - the recipe used by
heterogeneous driving simulators such as GPUDrive, Nocturne, Waymax, SMARTS and
MetaDrive. Only the type-specific pieces are overridden (`_build_simulator`,
`_observation`, spawn sampling via `_prepare_reset`, the per-type speed cap via
`_post_physics`, and per-type stats via `_augment_info`); everything else
(`_state`, `_dist_to_goal`, reward, `reset`/`step` skeleton, `render_frame`) is
inherited from the base env.

Four types, two kinematic families (via CompoundKinematicModel):
    vehicle, motorcycle, cyclist  -> KinematicBicycle  (action = accel, steer)
    pedestrian                    -> SimpleKinematicModel (omnidirectional)

Type-aware spawning: the spawn surface and routing graph are chosen per type
from lanelet2 participant traffic rules (``canPass``), which honour Autoware
participant tags + subtypes; headings follow the lanelet direction (maps are
mostly one_way); start positions are sampled along the lane and re-randomised
every reset; spawns are rejection-sampled to be collision-free. A one-hot type
code is appended to each observation. Reward is shared (inherited from the base):
    goal(+1) + progress - collision(0.5) - offroad(0.5).
"""
import numpy as np
import torch

from torchdrivesim.kinematic import KinematicBicycle, SimpleKinematicModel, CompoundKinematicModel
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.lanelet2 import load_lanelet_map
import lanelet2
from lanelet2.traffic_rules import Locations, Participants

from awsim_rl_env import AWSIMDrivingEnv
from awsim_lanelet2_traffic import (
    map_latlon_origin, build_driving_surface_mesh, build_route, mesh_camera,
    polyline_cumlen, point_at_arclen,
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
PARTICIPANT_ENUM = {
    "vehicle": "Vehicle", "motorcycle": "VehicleMotorcycle",
    "bicycle": "Bicycle", "pedestrian": "Pedestrian",
}


class AWSIMHeteroDrivingEnv(AWSIMDrivingEnv):
    ACT_DIM = 4                              # superset action (bicycle uses [:2], pedestrian [:4])
    EGO_DIM = AWSIMDrivingEnv.EGO_DIM + len(TYPES)              # ego features + own type one-hot
    PARTNER_FEATURES = AWSIMDrivingEnv.PARTNER_FEATURES + len(TYPES)  # + neighbour type one-hot
    OBS_DIM = (EGO_DIM + AWSIMDrivingEnv.MAX_PARTNERS * PARTNER_FEATURES
               + AWSIMDrivingEnv.MAX_ROAD * AWSIMDrivingEnv.ROAD_FEATURES)
    NUM_TYPES = len(TYPES)    # one policy head per road-user type
    # the ego type one-hot sits right after the base ego features
    TYPE_ONEHOT_SLICE = (AWSIMDrivingEnv.EGO_DIM, AWSIMDrivingEnv.EGO_DIM + len(TYPES))

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
        self._center, default_fov = mesh_camera(self.mesh)
        self.render_fov = render_fov if render_fov is not None else default_fov
        self._build_road_graph()

        # fixed per-agent type + derived static properties
        counts = self._allocate(num_agents, mix)
        self.agent_type_idx = np.concatenate([[i] * counts[t] for i, t in enumerate(TYPES)]).astype(int)
        self._rng.shuffle(self.agent_type_idx)
        self._type_idx_t = torch.tensor(self.agent_type_idx, device=device)
        self.model_assignments = torch.tensor(
            [[TYPE_SPEC[TYPES[t]]["model"] for t in self.agent_type_idx]], device=device)
        self.vmax = torch.tensor([TYPE_SPEC[TYPES[t]]["vmax"] for t in self.agent_type_idx],
                                 dtype=torch.float32, device=device)
        self.lr_all = torch.tensor([TYPE_SPEC[TYPES[t]]["lr"] for t in self.agent_type_idx],
                                   dtype=torch.float32, device=device)
        sizes = np.stack([TYPE_SPEC[TYPES[t]]["size"] for t in self.agent_type_idx])
        self.agent_size = torch.tensor(sizes, dtype=torch.float32, device=device).unsqueeze(0)
        self.type_onehot = torch.zeros(num_agents, len(TYPES), device=device)
        self.type_onehot[torch.arange(num_agents, device=device), self._type_idx_t] = 1.0

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
        """One pool of (polyline, arc-length cache) per participant, built once
        from that participant's traffic rules (canPass) and routing graph."""
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
                if poly.shape[0] >= 2:
                    cache = polyline_cumlen(poly)
                    if cache[2][-1] > 3.0:
                        polys.append((poly, cache))
            self._pools[pkey] = polys
        for pkey in participants:  # any empty pool falls back to the vehicle network
            if not self._pools[pkey]:
                self._pools[pkey] = self._pools.get("vehicle", [])

    def _sample_spawns(self):
        placed = []  # (x, y, radius)
        starts, headings, goals = [], [], []
        for tidx in self.agent_type_idx:
            spec = TYPE_SPEC[TYPES[tidx]]
            pool = self._pools[spec["participant"]]
            radius = 0.5 * max(spec["size"]) + self.spawn_gap
            chosen = None
            for _ in range(self.spawn_attempts):
                poly, cache = pool[self._rng.integers(len(pool))]
                total = cache[2][-1]
                if spec["goal_dist"] is None:            # pedestrian: cross the whole lanelet
                    forward = self._rng.random() < 0.5
                    s0, sg = (0.0, total) if forward else (total, 0.0)
                    x, y, h = point_at_arclen(poly, s0, cache)
                    if not forward:
                        h += np.pi
                else:                                    # wheeled: start along lane, goal ahead
                    s0 = self._rng.uniform(0.0, max(total - 3.0, 0.0) * 0.6)
                    sg = min(s0 + spec["goal_dist"], total)
                    x, y, h = point_at_arclen(poly, s0, cache)
                gx, gy, _ = point_at_arclen(poly, sg, cache)
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
        self._init_state[0, :, :2] = torch.tensor(np.stack(starts), dtype=torch.float32, device=self.device)
        self._init_state[0, :, 2] = torch.tensor(np.stack(headings), dtype=torch.float32, device=self.device)

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
            agent_types=self._type_idx_t.view(1, A), agent_type_names=TYPES,
        )

    # ---------------------------------------------------- overridden hooks
    def _prepare_reset(self):
        self._sample_spawns()  # re-randomise spawns every episode

    def _ego_features(self, state, prev_action):  # append own type one-hot
        return torch.cat([super()._ego_features(state, prev_action), self.type_onehot], dim=-1)

    def _partner_extra(self, nidx, valid):        # append each neighbour's type one-hot
        return self.type_onehot[nidx]

    def _post_physics(self):
        state = self._state()
        capped_v = torch.clamp(state[:, 3], -self.vmax, self.vmax)  # per-type top speed
        if not torch.equal(capped_v, state[:, 3]):
            new_state = state.clone(); new_state[:, 3] = capped_v
            self.simulator.set_state(new_state.unsqueeze(0))

    def _augment_info(self, info):
        for i, t in enumerate(TYPES):  # per-type goal-reaching
            m = (self._type_idx_t == i)
            info[f'reached_{t}'] = float(self._reached[m].float().mean()) if bool(m.any()) else 0.0
