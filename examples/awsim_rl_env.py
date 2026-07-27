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

from torchdrivesim._iou_utils import box2corners_th
from torchdrivesim.infractions import bbox2discs, point_to_mesh_distance_pt
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
    """Vectorised (num_agents parallel streams) driving env on one AWSIM map.

    Each agent observes its own ego state + goal AND its nearest neighbours, using
    the GPUDrive / Nocturne recipe: per-neighbour features in the ego frame
    (rel position, size, rel heading, signed speed), gathered for the K closest
    agents within a view radius and zero-padded. The policy encodes neighbours
    with a shared MLP and max-pools them (permutation-invariant), so the agent
    can see and react to others rather than only being punished after a collision.
    """

    ACT_DIM = 2
    EGO_DIM = 8               # ego features (speed, goal, prev action)
    MAX_PARTNERS = 8          # K nearest neighbours observed
    PARTNER_FEATURES = 7      # rel_x, rel_y, width, length, rel_head_cos/sin, rel_speed
    VIEW_RADIUS = 30.0        # metres; neighbours beyond this are not observed
    MAX_ROAD = 10             # K nearest road-graph points observed
    ROAD_FEATURES = 4         # rel_x, rel_y, rel_dir_cos, rel_dir_sin
    ROAD_RADIUS = 30.0        # metres; road points beyond this are not observed
    ROAD_POINT_CAP = 2000     # subsample the lane-centreline point cloud to this many
    OBS_DIM = EGO_DIM + MAX_PARTNERS * PARTNER_FEATURES + MAX_ROAD * ROAD_FEATURES
    NUM_TYPES = 1             # single agent type -> single policy head
    TYPE_ONEHOT_SLICE = None  # (start, end) of the ego type one-hot in the obs, if any

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
        self._build_road_graph()
        self._build_offroad_index()

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
        self._init_state[0, :, :2] = torch.tensor(starts, dtype=torch.float32, device=device)
        self._init_state[0, :, 2] = torch.tensor(headings, dtype=torch.float32, device=device)

        self.agent_length, self.agent_width, self.lr = 4.97, 2.04, 1.96
        self.agent_size = torch.tensor([self.agent_length, self.agent_width], device=device
                                       ).view(1, 1, 2).expand(1, num_agents, 2).contiguous()
        self._build_simulator()
        self.reset()

    def _build_road_graph(self):
        """Sample lane centrelines into a point cloud (x, y, heading) used for the
        road-graph observation, mirroring GPUDrive's road-segment observations."""
        xy, hdg = [], []
        for ll in self.lanelet_map.laneletLayer:
            pts = [(p.x, p.y) for p in ll.centerline]
            for a, b in zip(pts[:-1], pts[1:]):
                xy.append(a)
                hdg.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
        xy = np.asarray(xy, dtype=np.float32)
        hdg = np.asarray(hdg, dtype=np.float32)
        if xy.shape[0] > self.ROAD_POINT_CAP:  # subsample to bound the nearest-point query
            sel = self._rng.choice(xy.shape[0], self.ROAD_POINT_CAP, replace=False)
            xy, hdg = xy[sel], hdg[sel]
        self._road_xy = torch.tensor(xy, device=self.device)          # [M, 2]
        self._road_dir = torch.tensor(hdg, device=self.device)        # [M]

    # ------------------------------------------------------------- offroad
    # `Simulator.compute_offroad` (without pytorch3d) expands the *whole* driving
    # surface mesh once per agent corner and brute-forces the point-to-triangle
    # distance: cost and memory are O(num_agents x faces), which on the full
    # Shinjuku map (113k faces) is ~75% of the step time and OOMs past ~128 agents.
    # Only faces near an agent can be its closest face, so we keep the exact
    # distance computation but restrict it to the OFFROAD_FACES nearest faces
    # (by face centroid), which is a superset of the candidates by a wide margin.
    OFFROAD_FACES = 128

    def _build_offroad_index(self):
        verts = self.mesh.verts[0][..., :2]                            # [V, 2]
        faces = self.mesh.faces[0]                                     # [F, 3]
        tris = verts[faces]                                            # [F, 3, 2]
        self._face_tris = torch.nn.functional.pad(tris, (0, 1))        # [F, 3, 3], z=0
        self._face_centroid = tris.mean(dim=-2)                        # [F, 2]
        # circumradius per face: dist(agent, face) >= dist(agent, centroid) - radius,
        # so ranking by that lower bound (rather than by raw centroid distance) keeps
        # large triangles in the candidate set instead of losing them to nearer-centroid
        # small ones.
        self._face_radius = (tris - self._face_centroid.unsqueeze(-2)).norm(dim=-1).max(dim=-1).values
        self._offroad_k = min(self.OFFROAD_FACES, faces.shape[0])

    def _offroad(self, state):
        """Per-agent offroad loss, equivalent to `simulator.compute_offroad()[0]`."""
        A, K = self.num_agents, self._offroad_k
        lenwid = self.simulator.get_agent_size()[0][..., :2]           # [A, 2]
        rect = torch.cat([state[:, :2], lenwid, state[:, 2:3]], dim=-1)
        corners = box2corners_th(rect.unsqueeze(0))[0]                 # [A, 4, 2]
        # candidate faces: the K nearest by centroid to the agent centre
        d = torch.cdist(state[:, :2], self._face_centroid) - self._face_radius
        idx = d.topk(K, dim=-1, largest=False).indices                 # [A, K]
        tris = self._face_tris[idx]                                    # [A, K, 3, 3]
        tris = tris.unsqueeze(1).expand(A, 4, K, 3, 3).reshape(A * 4, K, 3, 3)
        pts = torch.nn.functional.pad(corners, (0, 1)).reshape(A * 4, 3)
        thr = self.simulator.cfg.offroad_threshold
        dist = point_to_mesh_distance_pt(pts, tris, threshold=thr)     # [A*4, 1]
        return dist.reshape(A, 4).sum(dim=-1) * self.simulator.get_present_mask()[0]

    # ----------------------------------------------------------- collision
    # `Simulator.compute_collision` loops over agents in Python (it carries a
    # "TODO: batch across agent dimension"), so at A agents it issues A x small
    # kernels per step and is launch-overhead bound. This is the same `discs`
    # metric computed for all agent pairs at once.
    def _collision(self, state):
        """Per-agent collision loss, equivalent to `simulator.compute_collision()[0]`."""
        A = self.num_agents
        size = self.simulator.get_agent_size()[0][..., :2]
        box = torch.nan_to_num(torch.cat([state[:, :2], size, state[:, 2:3]], dim=-1))
        centers, r = bbox2discs(box)                                   # [A, D, 2], [A, 1]
        D = centers.shape[-2]
        # map coordinates are O(100 m), so the matmul-based cdist loses precision
        # exactly where it matters (near-touching agents); ask for the direct form.
        flat = centers.reshape(A * D, 2)
        d = torch.cdist(flat, flat, compute_mode='donot_use_mm_for_euclid_dist')
        d = d.reshape(A, D, A, D).permute(0, 2, 1, 3).reshape(A, A, D * D)
        d = d.min(dim=-1).values                                       # [A, A] closest discs
        overlap = torch.relu(1 - d / (r + r.transpose(0, 1)))          # [A, A]
        mask = self.simulator.get_present_mask()[0].to(overlap.dtype)
        overlap = torch.nan_to_num(overlap) * mask.unsqueeze(0)
        overlap = overlap - torch.diag_embed(overlap.diagonal())       # drop self-overlap
        return overlap.sum(dim=-1) * mask

    def _build_simulator(self):
        A = self.num_agents
        kin = KinematicBicycle(dt=self.dt)
        kin.set_params(lr=torch.full((1, A), self.lr, device=self.device))
        kin.set_state(self._init_state.clone())
        # The action `_normalization_factor` is built on CPU in the constructor.
        kin = kin.to(self.device)
        cfg = TorchDriveConfig(left_handed_coordinates=False,
                               renderer=RendererConfig(left_handed_coordinates=False))
        self.renderer = renderer_from_config(cfg.renderer)
        self.simulator = Simulator(
            cfg=cfg, road_mesh=self.mesh, kinematic_model=kin, agent_size=self.agent_size,
            initial_present_mask=torch.ones(1, A, dtype=torch.bool, device=self.device),
            renderer=self.renderer, lanelet_map=[self.lanelet_map],
        )

    # ------------------------------------------------------------------ core
    def _state(self):
        return self.simulator.get_state()[0]  # (A, 4): x, y, psi, v

    def _dist_to_goal(self, state):
        return torch.linalg.norm(state[:, :2] - self.goals, dim=-1)  # (A,)

    def _observation(self, state, prev_action):
        # [A, EGO_DIM | MAX_PARTNERS*PARTNER_FEATURES | MAX_ROAD*ROAD_FEATURES];
        # the policy slices off the ego part and max-pools the partner and road sets.
        return torch.cat([self._ego_features(state, prev_action),
                          self._partner_block(state),
                          self._road_block(state)], dim=-1)

    def _ego_features(self, state, prev_action):
        x, y, psi, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        dx, dy = self.goals[:, 0] - x, self.goals[:, 1] - y
        c, s = torch.cos(psi), torch.sin(psi)
        gx_e = c * dx + s * dy            # goal in ego frame
        gy_e = -s * dx + c * dy
        dist = torch.linalg.norm(torch.stack([dx, dy], -1), dim=-1)
        head_err = torch.atan2(gy_e, gx_e)
        return torch.stack([
            v / 10.0,
            dist.clamp(max=100.0) / 50.0,
            torch.cos(head_err), torch.sin(head_err),
            (gx_e / 50.0).clamp(-2, 2), (gy_e / 50.0).clamp(-2, 2),
            prev_action[:, 0], prev_action[:, 1],
        ], dim=-1)

    def _partner_extra(self, nidx, valid):
        """Optional extra per-neighbour features (e.g. type one-hot). None in base."""
        return None

    def _partner_block(self, state):
        """GPUDrive-style neighbour observation: for each agent, the K nearest
        others within VIEW_RADIUS, in the ego frame, zero-padded. Returns
        [A, MAX_PARTNERS * PARTNER_FEATURES]."""
        A = state.shape[0]
        K = self.MAX_PARTNERS
        out = torch.zeros(A, K, self.PARTNER_FEATURES, device=self.device)
        k = min(K, A - 1)
        if k > 0:
            xy, psi, v = state[:, :2], state[:, 2], state[:, 3]
            dx = xy[:, 0][None, :] - xy[:, 0][:, None]          # [A, A] (j - i)
            dy = xy[:, 1][None, :] - xy[:, 1][:, None]
            dist2 = dx * dx + dy * dy
            dist2.fill_diagonal_(float('inf'))                  # ignore self
            nd, nidx = torch.topk(dist2, k, dim=1, largest=False)  # [A, k]
            valid = nd < self.VIEW_RADIUS ** 2
            c, s = torch.cos(psi)[:, None], torch.sin(psi)[:, None]
            gdx = torch.gather(dx, 1, nidx); gdy = torch.gather(dy, 1, nidx)
            rel_x = (c * gdx + s * gdy) / self.VIEW_RADIUS
            rel_y = (-s * gdx + c * gdy) / self.VIEW_RADIUS
            rel_h = psi[nidx] - psi[:, None]
            wj = self.agent_size[0, :, 1][nidx] / 3.0
            lj = self.agent_size[0, :, 0][nidx] / 6.0
            vj = v[nidx] / 10.0
            feat = torch.stack([rel_x, rel_y, wj, lj,
                                torch.cos(rel_h), torch.sin(rel_h), vj], dim=-1)  # [A, k, 7]
            extra = self._partner_extra(nidx, valid)
            if extra is not None:
                feat = torch.cat([feat, extra], dim=-1)
            feat = feat * valid[..., None]                      # zero out far/empty slots
            out[:, :k, :] = feat
        return out.reshape(A, -1)

    def _road_block(self, state):
        """GPUDrive-style road-graph observation: for each agent, the K nearest
        lane-centreline points within ROAD_RADIUS, in the ego frame, zero-padded.
        Returns [A, MAX_ROAD * ROAD_FEATURES]."""
        A = state.shape[0]
        K = self.MAX_ROAD
        out = torch.zeros(A, K, self.ROAD_FEATURES, device=self.device)
        M = self._road_xy.shape[0]
        k = min(K, M)
        if k > 0:
            psi = state[:, 2]
            dx = self._road_xy[:, 0][None, :] - state[:, 0][:, None]   # [A, M]
            dy = self._road_xy[:, 1][None, :] - state[:, 1][:, None]
            dist2 = dx * dx + dy * dy
            nd, nidx = torch.topk(dist2, k, dim=1, largest=False)      # [A, k]
            valid = nd < self.ROAD_RADIUS ** 2
            c, s = torch.cos(psi)[:, None], torch.sin(psi)[:, None]
            gdx = torch.gather(dx, 1, nidx); gdy = torch.gather(dy, 1, nidx)
            rel_x = (c * gdx + s * gdy) / self.ROAD_RADIUS
            rel_y = (-s * gdx + c * gdy) / self.ROAD_RADIUS
            rel_dir = self._road_dir[nidx] - psi[:, None]
            feat = torch.stack([rel_x, rel_y, torch.cos(rel_dir), torch.sin(rel_dir)], dim=-1)
            out[:, :k, :] = feat * valid[..., None]
        return out.reshape(A, -1)

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
        was_reached = self._reached.clone()          # agents that already finished before this step
        action = action.clone()
        action[was_reached] = 0.0                     # finished agents take no action (stay put)
        self.simulator.step(action.unsqueeze(0))
        self._post_physics()
        self._t += 1
        state = self._state()
        dist = self._dist_to_goal(state)

        progress = (self._prev_dist - dist)                                   # dense shaping
        collision = (self._collision(state) > 0).float()
        offroad = (self._offroad(state) > 0).float()
        newly_reached = (dist < self.goal_radius) & (~was_reached)
        reward = progress - 0.5 * collision - 0.5 * offroad + 1.0 * newly_reached.float()
        reward = torch.where(was_reached, torch.zeros_like(reward), reward)   # finished agents get 0

        self._reached = was_reached | (dist < self.goal_radius)
        # Freeze finished agents in place so they stop moving and don't drift into others.
        if self._reached.any():
            frozen = state.clone(); frozen[self._reached, 3] = 0.0
            self.simulator.set_state(frozen.unsqueeze(0), mask=self._reached.unsqueeze(0))
            state = self._state()
        self._prev_dist = dist
        self._prev_action = action
        done = self._reached.clone() | (self._t >= self.max_steps)
        # `active` marks transitions that count for training: an agent's steps are
        # valid up to and including the step it reaches the goal, then excluded.
        active = ~was_reached
        info = {
            'reached': float(self._reached.float().mean()),
            'collision': float(collision[active].mean()) if bool(active.any()) else 0.0,
            'offroad': float(offroad[active].mean()) if bool(active.any()) else 0.0,
            'active': active,
        }
        self._augment_info(info)
        return self._observation(state, action), reward, done, info

    # --------------------------------------------------------------- render
    def render_frame(self, follow=None, fov=None):
        """Bird's-eye frame. By default the camera frames the whole map, which on a
        1 km map makes the agents a few pixels wide; pass `follow` (an agent index)
        to centre on that agent and `fov` (metres across) to zoom in."""
        if follow is not None:
            centre = self._state()[follow, :2].tolist()
        else:
            centre = list(self._center)
        cam = torch.tensor([[centre]], device=self.device)
        psi = torch.zeros(1, 1, 1, device=self.device)
        fov = self.render_fov if fov is None else fov
        img = self.simulator.render(camera_xy=cam, camera_psi=psi,
                                    res=Resolution(self.render_res, self.render_res),
                                    fov=fov)
        return img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
