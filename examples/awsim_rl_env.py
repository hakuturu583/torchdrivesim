"""
A minimal multi-agent reinforcement-learning environment on an AWSIM / Autoware
Lanelet2 map, built on TorchDriveSim.

Design follows the GPUDrive / PufferDrive recipe used by
https://github.com/Emerge-Lab/Adaptive_Driving_Agent : a single shared policy
controls many "ego" vehicles at once, each with its own goal, using compact
ego-centric vector observations and a reward of
    goal(+1) + progress - collision(0.5) - offroad(0.5).

Every agent uses the kinematic bicycle model (action = normalised
acceleration, steering, both in [-1, 1]), so no mixed-model machinery is
needed and the whole batch of agents advances in one TorchDriveSim step.
Rendering is only used when saving a video, so rollouts stay fast on CPU.
"""
import numpy as np
import torch

from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.utils import Resolution

from awsim_lanelet2_traffic import (
    map_latlon_origin, build_driving_surface_mesh, build_route, _attr, mesh_camera,
)
from torchdrivesim.lanelet2 import load_lanelet_map
import lanelet2
from lanelet2.traffic_rules import Locations, Participants


class AWSIMDrivingEnv:
    """Vectorised (num_agents parallel streams) driving env on one AWSIM map."""

    OBS_DIM = 8
    ACT_DIM = 2

    def __init__(self, map_path, num_agents=8, max_steps=80, dt=0.1, device='cpu',
                 goal_radius=3.0, render_fov=None, render_res=512, seed=0):
        self.device = device
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.dt = dt
        self.goal_radius = goal_radius
        self.render_res = render_res
        self._rng = np.random.default_rng(seed)

        # --- map + mesh (Autoware projector semantics) ---
        origin = map_latlon_origin(map_path)
        self.lanelet_map = load_lanelet_map(map_path, origin=origin, robust=True,
                                            use_local_coordinates=True, recenter=True)
        self.mesh = build_driving_surface_mesh(self.lanelet_map).to(device)
        self._center, default_fov = mesh_camera(self.mesh)
        self.render_fov = render_fov if render_fov is not None else default_fov

        # --- lane-following routes -> start states + goals ---
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
        roads = sorted([l for l in self.lanelet_map.laneletLayer if _attr(l, 'subtype') == 'road'],
                       key=lambda l: l.id)
        routes, i = [], 0
        while len(routes) < num_agents and i < 4 * max(len(roads), 1):
            r = build_route(graph, roads[(i * 7) % len(roads)])
            if r.shape[0] >= 2 and np.hypot(*(r[-1] - r[0])) > 8.0:
                routes.append(r)
            i += 1
        if len(routes) < num_agents:  # pad by repeating if the map is small
            routes = (routes * (num_agents // max(len(routes), 1) + 1))[:num_agents]
        self.routes = routes

        starts = np.stack([r[0] for r in routes])          # (A, 2)
        headings = np.stack([np.arctan2(*(r[1] - r[0])[::-1]) for r in routes])  # (A,)
        self.goals = torch.tensor(np.stack([r[-1] for r in routes]), dtype=torch.float32, device=device)
        self._init_state = torch.zeros(1, num_agents, 4, device=device)
        self._init_state[0, :, :2] = torch.tensor(starts, dtype=torch.float32)
        self._init_state[0, :, 2] = torch.tensor(headings, dtype=torch.float32)

        self.agent_length, self.agent_width, self.lr = 4.97, 2.04, 1.96
        self._build_simulator()
        self.reset()

    def _build_simulator(self):
        A = self.num_agents
        agent_size = torch.tensor([self.agent_length, self.agent_width], device=self.device)
        agent_size = agent_size.view(1, 1, 2).expand(1, A, 2).contiguous()
        kin = KinematicBicycle(dt=self.dt)
        kin.set_params(lr=torch.full((1, A), self.lr, device=self.device))
        kin.set_state(self._init_state.clone())
        cfg = TorchDriveConfig(left_handed_coordinates=False,
                               renderer=RendererConfig(left_handed_coordinates=False))
        self.renderer = renderer_from_config(cfg.renderer)
        self.simulator = Simulator(
            cfg=cfg, road_mesh=self.mesh, kinematic_model=kin, agent_size=agent_size,
            initial_present_mask=torch.ones(1, A, dtype=torch.bool, device=self.device),
            renderer=self.renderer, lanelet_map=[self.lanelet_map],
        )

    # ------------------------------------------------------------------ core
    def _state(self):
        return self.simulator.get_state()[0]  # (A, 4): x, y, psi, v

    def _dist_to_goal(self, state):
        return torch.linalg.norm(state[:, :2] - self.goals, dim=-1)  # (A,)

    def _observation(self, state, prev_action):
        x, y, psi, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        dx, dy = self.goals[:, 0] - x, self.goals[:, 1] - y
        c, s = torch.cos(psi), torch.sin(psi)
        gx_e = c * dx + s * dy            # goal in ego frame
        gy_e = -s * dx + c * dy
        dist = torch.linalg.norm(torch.stack([dx, dy], -1), dim=-1)
        head_err = torch.atan2(gy_e, gx_e)
        obs = torch.stack([
            v / 10.0,
            dist.clamp(max=100.0) / 50.0,
            torch.cos(head_err), torch.sin(head_err),
            (gx_e / 50.0).clamp(-2, 2), (gy_e / 50.0).clamp(-2, 2),
            prev_action[:, 0], prev_action[:, 1],
        ], dim=-1)
        return obs

    # Hooks so subclasses (e.g. the heterogeneous env) can extend reset/step
    # without duplicating their bodies.
    def _prepare_reset(self):
        """Called at the start of reset(); override to re-sample spawns etc."""

    def _post_physics(self):
        """Called after simulator.step(); override e.g. to clamp per-type speed."""

    def _augment_info(self, info):
        """Called at the end of step(); override to add extra info entries."""

    def reset(self):
        self._prepare_reset()
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
        self._post_physics()
        self._t += 1
        state = self._state()
        dist = self._dist_to_goal(state)

        progress = (self._prev_dist - dist)                                   # dense shaping
        collision = (self.simulator.compute_collision()[0] > 0).float()
        offroad = (self.simulator.compute_offroad()[0] > 0).float()
        newly_reached = (dist < self.goal_radius) & (~self._reached)
        reward = progress - 0.5 * collision - 0.5 * offroad + 1.0 * newly_reached.float()
        reward = torch.where(self._reached, torch.zeros_like(reward), reward)  # frozen agents get 0

        self._reached |= (dist < self.goal_radius)
        self._prev_dist = dist
        self._prev_action = action
        done = self._reached.clone() | (self._t >= self.max_steps)
        info = {
            'reached': float(self._reached.float().mean()),
            'collision': float(collision.mean()),
            'offroad': float(offroad.mean()),
        }
        self._augment_info(info)
        return self._observation(state, action), reward, done, info

    # --------------------------------------------------------------- render
    def render_frame(self):
        cam = torch.tensor([[list(self._center)]], device=self.device)
        psi = torch.zeros(1, 1, 1, device=self.device)
        img = self.simulator.render(camera_xy=cam, camera_psi=psi,
                                    res=Resolution(self.render_res, self.render_res),
                                    fov=self.render_fov)
        return img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
