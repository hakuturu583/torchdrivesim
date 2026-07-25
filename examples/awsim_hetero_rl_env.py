"""
Heterogeneous multi-agent RL environment on an AWSIM / Autoware map.

Extends awsim_rl_env with several road-user *types* trained together by one
shared, type-conditioned policy - the recipe used by heterogeneous driving
simulators such as GPUDrive, Nocturne, Waymax, SMARTS and MetaDrive, which all
mix vehicles, cyclists and pedestrians in a single scene.

Four types, two kinematic families (via CompoundKinematicModel):
    vehicle, motorcycle, cyclist  -> KinematicBicycle  (action = accel, steer)
    pedestrian                    -> SimpleKinematicModel (omnidirectional)
Types differ in size, turn radius, top speed, spawn surface (road lanelets for
wheeled agents, crosswalk lanelets for pedestrians) and colour. A one-hot type
code is appended to each agent's observation so the shared policy can special-
ise its behaviour per type. Reward is the same for all:
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

# type -> (kinematic family index, size L, W, lr, top speed m/s, colour RGB)
TYPES = ["vehicle", "motorcycle", "cyclist", "pedestrian"]
TYPE_SPEC = {
    "vehicle":    dict(model=0, size=(4.97, 2.04), lr=1.96, vmax=14.0, color=(32, 74, 135)),
    "motorcycle": dict(model=0, size=(2.20, 0.90), lr=0.80, vmax=18.0, color=(230, 90, 20)),
    "cyclist":    dict(model=0, size=(1.80, 0.70), lr=0.60, vmax=6.0,  color=(24, 104, 225)),
    "pedestrian": dict(model=1, size=(0.70, 0.70), lr=0.50, vmax=2.0,  color=(173, 127, 168)),
}
DEFAULT_MIX = {"vehicle": 0.4, "motorcycle": 0.15, "cyclist": 0.15, "pedestrian": 0.3}


class AWSIMHeteroDrivingEnv:
    OBS_DIM = 8 + len(TYPES)  # base features + one-hot type
    ACT_DIM = 4               # superset action (bicycle uses [:2], pedestrian [:4])

    def __init__(self, map_path, num_agents=12, max_steps=80, dt=0.1, device='cpu',
                 goal_radius=3.0, mix=None, render_fov=None, render_res=512, seed=0):
        self.device = device
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.dt = dt
        self.goal_radius = goal_radius
        self.render_res = render_res
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

        # --- assign a type to each agent ---
        counts = self._allocate(num_agents, mix)
        self.agent_type_idx = np.concatenate([[i] * counts[t] for i, t in enumerate(TYPES)]).astype(int)
        self._rng.shuffle(self.agent_type_idx)

        # --- routes / goals per type (road lanelets vs crosswalks) ---
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
        road_ll = sorted([l for l in self.lanelet_map.laneletLayer if _attr(l, 'subtype') == 'road'],
                         key=lambda l: l.id)
        walk_ll = sorted([l for l in self.lanelet_map.laneletLayer
                          if _attr(l, 'subtype') in ('crosswalk', 'road_shoulder')], key=lambda l: l.id)
        starts, headings, goals = [], [], []
        for k, tidx in enumerate(self.agent_type_idx):
            is_ped = TYPES[tidx] == "pedestrian"
            route = self._pedestrian_route(walk_ll, k) if is_ped else self._vehicle_route(graph, road_ll, k)
            starts.append(route[0])
            headings.append(np.arctan2(*(route[1] - route[0])[::-1]))
            goals.append(route[-1])
        self.goals = torch.tensor(np.stack(goals), dtype=torch.float32, device=device)
        self._init_state = torch.zeros(1, num_agents, 4, device=device)
        self._init_state[0, :, :2] = torch.tensor(np.stack(starts), dtype=torch.float32)
        self._init_state[0, :, 2] = torch.tensor(np.stack(headings), dtype=torch.float32)

        # per-agent static properties
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

        self._build_simulator()
        self.reset()

    # ----------------------------------------------------------- construction
    def _allocate(self, n, mix):
        counts = {t: int(np.floor(n * mix.get(t, 0))) for t in TYPES}
        while sum(counts.values()) < n:  # give remainder to vehicles
            counts["vehicle"] += 1
        return counts

    def _vehicle_route(self, graph, road_ll, k):
        for j in range(len(road_ll)):
            r = build_route(graph, road_ll[(k * 7 + j) % len(road_ll)])
            if r.shape[0] >= 2 and np.hypot(*(r[-1] - r[0])) > 8.0:
                return r
        return build_route(graph, road_ll[k % len(road_ll)])

    def _pedestrian_route(self, walk_ll, k):
        if walk_ll:
            ll = walk_ll[k % len(walk_ll)]
            pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
            if pts.shape[0] >= 2 and np.hypot(*(pts[-1] - pts[0])) > 1.0:
                return pts
        # fallback: a short straight crossing near the map centre
        c = np.array(self._center)
        off = self._rng.uniform(-30, 30, size=2)
        return np.stack([c + off, c + off + self._rng.uniform(-10, 10, size=2)])

    def _build_simulator(self):
        A = self.num_agents
        assign = self.model_assignments
        wheeled = (assign[0] == 0)
        ped = (assign[0] == 1)
        bike = KinematicBicycle(dt=self.dt)
        bike.set_params(lr=self.lr_all[wheeled].clone())
        bike.set_state(self._init_state[0, wheeled].clone())
        walk = SimpleKinematicModel(dt=self.dt, max_dx=TYPE_SPEC["pedestrian"]["vmax"])
        walk.set_state(self._init_state[0, ped].clone())
        kin = CompoundKinematicModel([bike, walk], model_assignments=assign, dt=self.dt)

        renderer = renderer_from_config(RendererConfig(left_handed_coordinates=False))
        for t in TYPES:  # register per-type colours + rendering levels
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
        # enforce per-type top speed for wheeled agents
        state = self._state()
        capped_v = torch.clamp(state[:, 3], -self.vmax, self.vmax)
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
        for i, t in enumerate(TYPES):  # per-type goal-reaching
            m = (ti == i)
            info[f'reached_{t}'] = float(self._reached[m].float().mean()) if m.any() else 0.0
        return self._observation(state, action), reward, done, info

    def render_frame(self):
        cam = torch.tensor([[list(self._center)]], device=self.device)
        psi = torch.zeros(1, 1, 1, device=self.device)
        img = self.simulator.render(camera_xy=cam, camera_psi=psi,
                                    res=Resolution(self.render_res, self.render_res), fov=self.render_fov)
        return img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
