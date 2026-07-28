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
    polyline_cumlen, point_at_arclen,
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
    EGO_DIM = 14              # speed, goal, prev action, signal (5), speed limit
    MAX_PARTNERS = 8          # K nearest neighbours observed
    PARTNER_FEATURES = 7      # rel_x, rel_y, width, length, rel_head_cos/sin, rel_speed
    VIEW_RADIUS = 30.0        # metres; neighbours beyond this are not observed
    MAX_ROAD = 10             # K nearest road-graph points observed
    ROAD_FEATURES = 4         # rel_x, rel_y, rel_dir_cos, rel_dir_sin
    ROAD_RADIUS = 30.0        # metres; road points beyond this are not observed
    ROAD_POINT_CAP = 2000     # subsample the lane-centreline point cloud to this many
                              # (observation only - rules use the dense table below)
    OBS_DIM = EGO_DIM + MAX_PARTNERS * PARTNER_FEATURES + MAX_ROAD * ROAD_FEATURES
    NUM_TYPES = 1             # single agent type -> single policy head
    TYPE_ONEHOT_SLICE = None  # (start, end) of the ego type one-hot in the obs, if any

    # Reward weights. `progress` is measured in metres, so it sums to roughly the goal
    # distance over a successful episode (~45 m for a vehicle): a goal bonus of 1.0 is
    # then worth 2% of the return and the policy optimises "make progress", not "arrive".
    # Likewise an offroad penalty has to outweigh the progress won by cutting a corner.
    # Prefer scaling `progress` down over scaling the goal bonus up: the latter inflates
    # the return scale, and with vf_coef=2.0 the value loss then dominates the gradient.
    W_PROGRESS = 1.0
    W_GOAL = 1.0
    W_OFFROAD = 0.5
    W_COLLISION = 0.5

    # Rolling goals: instead of parking an agent at its single goal, the next goal is
    # placed GOAL_DIST further along the same route, and the route is extended through
    # the lanelet graph when it runs out. An episode is then a continuous drive ending
    # only at max_steps, and the score is "goals collected", not "did it arrive".
    GOAL_DIST = 150.0
    ROLLING_GOALS = True

    # Traffic rules read off the map: per-lanelet speed limits, and the stop lines of
    # traffic_light regulatory elements. Every rule below is BOTH observed (in
    # _ego_features) and rewarded - a penalty for something the agent cannot perceive
    # is not a hard task, it is an unlearnable one.
    DEFAULT_SPEED_LIMIT = 50 / 3.6   # m/s, for lanelets with no speed_limit tag
    TL_CYCLE = 20.0                  # seconds per full signal cycle
    TL_RADIUS = 60.0                 # metres; stop lines beyond this are not observed
    INTERSECTION_RADIUS = 40.0       # metres; stop lines within this share a junction
    W_REDLIGHT = 0.5                 # penalty for crossing a stop line on red
    W_WRONGWAY = 0.2                 # penalty per step against the lane direction
    # Speeding: the penalty is w * (v - limit) / limit and the progress it buys is
    # w_progress * (v - limit) * dt, so compliance beats speeding by
    #     w_speeding / (w_progress * dt * limit)
    # - independent of how far over the agent is, and *weakest on fast roads*. At the
    # map's most common 50 km/h that ratio is only 1.4 with w=0.2, which PPO reads as a
    # near-tie; 0.6 makes it ~4.3.
    W_SPEEDING = 0.6
    # vehicle vmax (14.0 m/s) sits a hair above the 50 km/h limit (13.89), so counting
    # any excess at all reports a full-speed car on a main road as a violation. Only
    # count a real margin, and report the magnitude separately.
    SPEEDING_TOLERANCE = 0.05

    def __init__(self, map_path, num_agents=8, max_steps=80, dt=0.1, device='cpu',
                 goal_radius=3.0, render_fov=None, render_res=512, seed=0,
                 w_progress=None, w_goal=None, w_offroad=None, w_collision=None,
                 rolling_goals=None, goal_dist=None,
                 w_redlight=None, w_wrongway=None, w_speeding=None):
        self.w_redlight = self.W_REDLIGHT if w_redlight is None else w_redlight
        self.w_wrongway = self.W_WRONGWAY if w_wrongway is None else w_wrongway
        self.w_speeding = self.W_SPEEDING if w_speeding is None else w_speeding
        self.rolling_goals = self.ROLLING_GOALS if rolling_goals is None else rolling_goals
        self.default_goal_dist = self.GOAL_DIST if goal_dist is None else goal_dist
        self.w_progress = self.W_PROGRESS if w_progress is None else w_progress
        self.w_goal = self.W_GOAL if w_goal is None else w_goal
        self.w_offroad = self.W_OFFROAD if w_offroad is None else w_offroad
        self.w_collision = self.W_COLLISION if w_collision is None else w_collision
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
        self._build_rule_lookup()
        self._build_traffic_lights()
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
        self._init_route_state(routes, np.zeros(num_agents),
                               np.full(num_agents, self.default_goal_dist))
        self._init_state = torch.zeros(1, num_agents, 4, device=device)
        self._init_state[0, :, :2] = torch.tensor(starts, dtype=torch.float32, device=device)
        self._init_state[0, :, 2] = torch.tensor(headings, dtype=torch.float32, device=device)

        self.agent_length, self.agent_width, self.lr = 4.97, 2.04, 1.96
        self.agent_size = torch.tensor([self.agent_length, self.agent_width], device=device
                                       ).view(1, 1, 2).expand(1, num_agents, 2).contiguous()
        self._build_simulator()
        self.reset()

    def _build_road_graph(self):
        """Sample lane centrelines into a point cloud (x, y, heading, speed limit) used
        for the road-graph observation, mirroring GPUDrive's road-segment observations.
        The heading also gives the legal direction of travel (wrong-way detection) and
        the speed limit the legal speed, both looked up via the nearest point."""
        xy, hdg, lim = [], [], []
        for ll in self.lanelet_map.laneletLayer:
            # every lanelet in the AWSIM map carries speed_limit, in km/h
            v = float(ll.attributes['speed_limit']) / 3.6 if 'speed_limit' in ll.attributes \
                else self.DEFAULT_SPEED_LIMIT
            pts = [(p.x, p.y) for p in ll.centerline]
            for a, b in zip(pts[:-1], pts[1:]):
                xy.append(a)
                hdg.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
                lim.append(v)
        xy = np.asarray(xy, dtype=np.float32)
        hdg = np.asarray(hdg, dtype=np.float32)
        lim = np.asarray(lim, dtype=np.float32)
        if xy.shape[0] > self.ROAD_POINT_CAP:  # subsample to bound the nearest-point query
            sel = self._rng.choice(xy.shape[0], self.ROAD_POINT_CAP, replace=False)
            xy, hdg, lim = xy[sel], hdg[sel], lim[sel]
        self._road_xy = torch.tensor(xy, device=self.device)          # [M, 2]
        self._road_dir = torch.tensor(hdg, device=self.device)        # [M]
        self._road_speed = torch.tensor(lim, device=self.device)      # [M] m/s

    def _build_rule_lookup(self):
        """Dense, per-participant lane-point table backing the speed-limit and wrong-way
        rules.

        The observation cloud above is subsampled to ROAD_POINT_CAP, which is fine for
        perception but not for a penalty: at 2000 points over 41.7 km of lane the nearest
        point is 21 m away, and at spawn - agents sitting exactly on a centreline, facing
        the right way - 28% of them read as wrong-way and 20% pick up another lanelet's
        speed limit. A penalty that fires on a quarter of correctly driving agents is
        noise, not a training signal. This table is therefore not subsampled, and it is
        masked per participant, because pedestrians on a crosswalk otherwise match the
        road lanelet crossing underneath them (their false wrong-way rate is 0.28 against
        all lanelets, 0.01 against walkways and crosswalks).
        """
        xy, hdg, lim, lls = [], [], [], []
        for k, ll in enumerate(self.lanelet_map.laneletLayer):
            v = float(ll.attributes['speed_limit']) / 3.6 if 'speed_limit' in ll.attributes \
                else self.DEFAULT_SPEED_LIMIT
            pts = [(p.x, p.y) for p in ll.centerline]
            for a, b in zip(pts[:-1], pts[1:]):
                xy.append(a)
                hdg.append(np.arctan2(b[1] - a[1], b[0] - a[0]))
                lim.append(v)
                lls.append(ll)
        self._rule_xy = torch.tensor(np.asarray(xy, dtype=np.float32), device=self.device)
        self._rule_dir = torch.tensor(np.asarray(hdg, dtype=np.float32), device=self.device)
        self._rule_speed = torch.tensor(np.asarray(lim, dtype=np.float32), device=self.device)
        self._rule_allowed = self._rule_participant_mask(lls)

    def _rule_participant_mask(self, lanelets):
        """[NUM_TYPES, M] mask of which lane points each agent type may be matched to.
        The base env has a single vehicle type, so everything is allowed."""
        return torch.ones(self.NUM_TYPES, len(lanelets), dtype=torch.bool, device=self.device)

    def _agent_types(self):
        """Per-agent index into the first dimension of `_rule_allowed`."""
        return torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

    # -------------------------------------------------------- traffic lights
    def _build_traffic_lights(self):
        """Stop lines of the map's traffic_light regulatory elements, grouped into
        intersections and split into two alternating phase groups.

        Caveat: the map stores the geometry, not a signal plan, so the phases here are
        synthetic - a fixed cycle per intersection with the two roughly perpendicular
        approach groups in antiphase. That is enough to require stopping and to make
        crossing traffic mutually exclusive, but it is not Autoware's real signal logic.
        """
        seen, sig = {}, []
        for ll in self.lanelet_map.laneletLayer:
            for r in ll.regulatoryElements:
                if 'subtype' not in r.attributes or r.attributes['subtype'] != 'traffic_light':
                    continue
                if r.id in seen:
                    continue
                try:
                    line = r.stopLine
                    if line is None:
                        continue
                    pts = [(q.x, q.y) for q in line]
                except Exception:
                    continue
                if len(pts) < 1:
                    continue
                mid = np.mean(np.asarray(pts, dtype=np.float64), axis=0)
                cl = [(q.x, q.y) for q in ll.centerline]           # approach direction
                if len(cl) < 2:
                    continue
                a, b = np.asarray(cl[-2]), np.asarray(cl[-1])
                seen[r.id] = len(sig)
                sig.append((mid[0], mid[1], float(np.arctan2(b[1] - a[1], b[0] - a[0]))))
        if not sig:
            self._sl_xy = torch.zeros(0, 2, device=self.device)
            self._sl_dir = torch.zeros(0, device=self.device)
            self._sl_group = torch.zeros(0, dtype=torch.long, device=self.device)
            self._sl_offset = torch.zeros(0, dtype=torch.long, device=self.device)
            return
        sig = np.asarray(sig, dtype=np.float64)

        # greedy spatial clustering of stop lines into intersections
        cluster, centres = np.full(len(sig), -1), []
        for i, (x, y, _) in enumerate(sig):
            for c, (cx, cy) in enumerate(centres):
                if (x - cx) ** 2 + (y - cy) ** 2 < self.INTERSECTION_RADIUS ** 2:
                    cluster[i] = c
                    break
            else:
                cluster[i] = len(centres)
                centres.append((x, y))

        # within an intersection, approaches roughly parallel to the first one share a
        # phase; the roughly perpendicular ones get the opposite phase
        group = np.zeros(len(sig), dtype=np.int64)
        offset = np.zeros(len(sig), dtype=np.int64)
        period = max(int(round(self.TL_CYCLE / self.dt)), 2)
        for c in range(len(centres)):
            members = np.flatnonzero(cluster == c)
            ref = sig[members[0], 2]
            group[members] = (np.abs(np.cos(sig[members, 2] - ref)) < 0.5).astype(np.int64)
            offset[members] = int(self._rng.integers(period))   # desynchronise junctions
        self._sl_xy = torch.tensor(sig[:, :2], dtype=torch.float32, device=self.device)
        self._sl_dir = torch.tensor(sig[:, 2], dtype=torch.float32, device=self.device)
        self._sl_group = torch.tensor(group, device=self.device)
        self._sl_offset = torch.tensor(offset, device=self.device)

    def _light_state(self):
        """0 = red, 1 = amber, 2 = green, for every stop line at the current step."""
        period = max(int(round(self.TL_CYCLE / self.dt)), 2)
        phase = (self._t + self._sl_offset + self._sl_group * (period // 2)) % period
        green = (phase < int(period * 0.42))
        amber = (~green) & (phase < period // 2)
        return torch.where(green, 2, torch.where(amber, 1, 0))

    def _signals(self, state):
        """For each agent, the stop line it is approaching (if any) within TL_RADIUS:
        returns (index or -1, signed distance along the agent's heading, features)."""
        A = state.shape[0]
        if self._sl_xy.shape[0] == 0:
            return (torch.full((A,), -1, dtype=torch.long, device=self.device),
                    torch.zeros(A, device=self.device), torch.zeros(A, 5, device=self.device))
        psi = state[:, 2]
        fwd = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)     # [A, 2]
        rel = self._sl_xy.unsqueeze(0) - state[:, :2].unsqueeze(1)      # [A, S, 2]
        ahead = (rel * fwd.unsqueeze(1)).sum(-1)                        # [A, S] signed
        dist = rel.norm(dim=-1)
        # only a stop line in front, within range, and facing the way the agent drives
        aligned = torch.cos(psi.unsqueeze(1) - self._sl_dir.unsqueeze(0)) > 0.5
        valid = aligned & (ahead > 0) & (dist < self.TL_RADIUS)
        pick = torch.where(valid, dist, torch.full_like(dist, float('inf'))).min(dim=1)
        has = torch.isfinite(pick.values)
        idx = torch.where(has, pick.indices, torch.full_like(pick.indices, -1))
        signed = torch.where(has, ahead.gather(1, pick.indices.unsqueeze(1)).squeeze(1),
                             torch.zeros(A, device=self.device))
        colour = self._light_state()[pick.indices.clamp(min=0)]
        onehot = torch.nn.functional.one_hot(colour, 3).float() * has.unsqueeze(1).float()
        feat = torch.cat([has.float().unsqueeze(1),
                          (signed / self.TL_RADIUS).clamp(0, 1).unsqueeze(1), onehot], dim=-1)
        return idx, signed, feat

    def _rule_violations(self, state):
        """Red-light crossings, wrong-way driving and speeding, as [A] tensors.

        A red-light violation is the moment an agent passes the stop line it was
        approaching while that light is red, so it is detected against the stop line
        selected on the *previous* step - once crossed, the stop line is behind the
        agent and is no longer selected.
        """
        A = state.shape[0]
        psi = state[:, 2]
        fwd = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)
        idx, signed, _ = self._signals(state)

        redlight = torch.zeros(A, device=self.device)
        if self._sl_xy.shape[0] > 0:
            prev = self._prev_sl_idx
            had = prev >= 0
            pidx = prev.clamp(min=0)
            now_signed = ((self._sl_xy[pidx] - state[:, :2]) * fwd).sum(-1)
            crossed = had & (self._prev_sl_signed > 0) & (now_signed <= 0)
            redlight = (crossed & (self._light_state()[pidx] == 0)).float()
        self._prev_sl_idx, self._prev_sl_signed = idx, signed

        near = self._nearest_rule_point(state)
        # heading against the lane direction by more than 90 degrees
        wrongway = (torch.cos(psi - self._rule_dir[near]) < 0).float()
        limit = self._rule_speed[near]
        speeding = (state[:, 3].abs() - limit).clamp(min=0) / limit
        return redlight, wrongway, speeding

    def _nearest_rule_point(self, state):
        """Nearest lane point for each agent, restricted to the ones its participant
        type may legally be on."""
        d = torch.cdist(state[:, :2], self._rule_xy)
        d = d.masked_fill(~self._rule_allowed[self._agent_types()], float('inf'))
        return d.argmin(dim=1)

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

    # ------------------------------------------------------- rolling goals
    def _init_route_state(self, polys, s_start, goal_dist):
        """Per-agent route bookkeeping: the polyline the agent follows, its arc-length
        cache, and the arc-length at which its current goal sits."""
        self._route_poly = list(polys)
        self._route_cache = [polyline_cumlen(p) for p in self._route_poly]
        self._goal_dist = np.asarray(goal_dist, dtype=np.float64)
        self._s_goal = np.minimum(np.asarray(s_start, dtype=np.float64) + self._goal_dist,
                                  self._route_totals())
        self._s_goal0 = self._s_goal.copy()   # restored by reset()
        self._goals_np = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._route_exhausted = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._goals_reached = torch.zeros(self.num_agents, device=self.device)
        self._sync_goals(range(self.num_agents))

    def _reset_route_state(self):
        """Rewind the rolling goals to where the current spawn placed them."""
        self._s_goal = self._s_goal0.copy()
        self._sync_goals(range(self.num_agents))

    def _route_totals(self):
        return np.array([float(c[2][-1]) for c in self._route_cache])

    def _sync_goals(self, idx):
        """Refresh the goal positions of `idx` from their route arc-lengths. Goals live
        in a numpy buffer so only one host-to-device copy happens per step."""
        for i in idx:
            self._goals_np[i] = point_at_arclen(self._route_poly[i], self._s_goal[i],
                                                self._route_cache[i])[:2]
        self.goals = torch.as_tensor(self._goals_np, device=self.device)

    def _extend_route(self, i):
        """Chain more lanelets onto agent i's route. False if it cannot be extended;
        the base env's routes are fixed, so only the hetero env implements this."""
        return False

    def _respawn(self, i):
        """Move agent i to a fresh route after it runs out of road. Returns the new
        (x, y, heading), or None if the env cannot respawn (base env)."""
        return None

    def _advance_goals(self, mask):
        """Place the next goal further along the route for every agent in `mask`.

        The map is a finite cut-out, so a route eventually reaches its edge; rather
        than parking the agent there, it is respawned on a fresh route (which is why
        this returns whether any agent moved - the caller must re-read the state).
        """
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        moved = []
        for i in idx:
            s = self._s_goal[i] + self._goal_dist[i]
            total = float(self._route_cache[i][2][-1])
            if s > total - 1e-3:
                if self._extend_route(i):
                    total = float(self._route_cache[i][2][-1])
                else:
                    pose = self._respawn(i)
                    if pose is None:   # nowhere to go: the agent stops earning goals
                        self._route_exhausted[i] = True
                        self._s_goal[i] = total
                        continue
                    moved.append((i, pose))
                    total = float(self._route_cache[i][2][-1])
                    s = min(self._goal_dist[i], total)
            self._s_goal[i] = min(s, total)
        self._sync_goals(idx)
        if moved:
            state = self._state().clone()
            m = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
            for i, (x, y, h) in moved:
                state[i, 0], state[i, 1], state[i, 2], state[i, 3] = x, y, h, 0.0
                m[i] = True
                self._prev_sl_idx[i] = -1
            self.simulator.set_state(state.unsqueeze(0), mask=m.unsqueeze(0))
        return bool(moved)

    def _observation(self, state, prev_action):
        # [A, EGO_DIM | MAX_PARTNERS*PARTNER_FEATURES | MAX_ROAD*ROAD_FEATURES];
        # the policy slices off the ego part and max-pools the partner and road sets.
        return torch.cat([self._ego_features(state, prev_action),
                          self._partner_block(state),
                          self._road_block(state)], dim=-1)

    def _ego_features(self, state, prev_action):
        x, y, psi, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        limit = self._rule_speed[self._nearest_rule_point(state)]
        dx, dy = self.goals[:, 0] - x, self.goals[:, 1] - y
        c, s = torch.cos(psi), torch.sin(psi)
        gx_e = c * dx + s * dy            # goal in ego frame
        gy_e = -s * dx + c * dy
        dist = torch.linalg.norm(torch.stack([dx, dy], -1), dim=-1)
        head_err = torch.atan2(gy_e, gx_e)
        base = torch.stack([
            v / 10.0,
            dist.clamp(max=100.0) / 50.0,
            torch.cos(head_err), torch.sin(head_err),
            (gx_e / 50.0).clamp(-2, 2), (gy_e / 50.0).clamp(-2, 2),
            prev_action[:, 0], prev_action[:, 1],
            limit / 10.0,
        ], dim=-1)
        # the signal block is what makes the red-light penalty learnable
        return torch.cat([base, self._signals(state)[2]], dim=-1)

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
        self._reset_route_state()
        self._build_simulator()
        self._t = 0
        self._prev_action = torch.zeros(self.num_agents, self.ACT_DIM, device=self.device)
        self._reached = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._goals_reached = torch.zeros(self.num_agents, device=self.device)
        self._route_exhausted = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._prev_sl_idx = torch.full((self.num_agents,), -1, dtype=torch.long, device=self.device)
        self._prev_sl_signed = torch.zeros(self.num_agents, device=self.device)
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
        redlight, wrongway, speeding = self._rule_violations(state)
        newly_reached = (dist < self.goal_radius) & (~was_reached)
        reward = (self.w_progress * progress - self.w_collision * collision
                  - self.w_offroad * offroad + self.w_goal * newly_reached.float()
                  - self.w_redlight * redlight - self.w_wrongway * wrongway
                  - self.w_speeding * speeding)
        reward = torch.where(was_reached, torch.zeros_like(reward), reward)   # finished agents get 0

        self._goals_reached += newly_reached.float()
        if self.rolling_goals:
            # Hand out the next goal instead of parking the agent. Only an agent whose
            # route cannot be extended any further is finished.
            if bool(newly_reached.any()):
                if self._advance_goals(newly_reached):
                    state = self._state()      # some agents were respawned elsewhere
                # the goal moved, so re-measure: otherwise the jump in distance would be
                # charged to the next step as a large negative `progress`
                dist = self._dist_to_goal(state)
            self._reached = was_reached | (newly_reached & self._route_exhausted)
        else:
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
            'goals': float(self._goals_reached.mean()),   # goals collected per agent
            'redlight': float(redlight.sum()),            # crossings on red this step
            'wrongway': float(wrongway[active].mean()) if bool(active.any()) else 0.0,
            'speeding': float((speeding[active] > self.SPEEDING_TOLERANCE).float().mean())
                        if bool(active.any()) else 0.0,
            'speed_excess': float(speeding[active].mean()) if bool(active.any()) else 0.0,
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
