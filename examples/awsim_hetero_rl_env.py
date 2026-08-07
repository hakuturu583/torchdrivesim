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
    pedestrian                    -> OrientedKinematicModel (omnidirectional,
                                     body-frame action to match the ego-frame observation)

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

from torchdrivesim.kinematic import (KinematicBicycle, OrientedKinematicModel,
                                     CompoundKinematicModel)
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.lanelet2 import load_lanelet_map
import lanelet2
from lanelet2.traffic_rules import Locations, Participants

from awsim_rl_env import AWSIMDrivingEnv, SteeringLimitedBicycle
from awsim_lanelet2_traffic import (
    map_latlon_origin, build_driving_surface_mesh, build_route, mesh_camera,
    polyline_cumlen, point_at_arclen,
)

# type -> kinematic family, size, turn radius, top speed, lanelet2 participant,
# goal distance, colour. The participant selects which lanelets the type may
# spawn/route on (via traffic_rules.canPass) and which routing graph is used,
# so Autoware participant tags + subtypes are honoured directly.
TYPES = ["vehicle", "motorcycle", "cyclist", "pedestrian"]
# `max_steer` is the slip angle one unit of steering action buys, in radians. It has to
# be per type: the heading rate is v / lr * sin(beta), so the single library-default
# pi/2 gave a motorcycle (lr = 0.8 m) three times a car's turn rate for the same command
# and it learned to spin. The speed-dependent cap in AWSIMDrivingEnv.LAT_ACCEL_MAX sits
# on top of this one; these are the geometric limits, roughly the steering lock of each.
TYPE_SPEC = {
    "vehicle":    dict(model=0, size=(4.97, 2.04), lr=1.96, vmax=14.0, max_steer=0.30, participant="vehicle",    goal_dist=150.0, color=(32, 74, 135)),
    "motorcycle": dict(model=0, size=(2.20, 0.90), lr=0.80, vmax=18.0, max_steer=0.35, participant="motorcycle", goal_dist=150.0, color=(230, 90, 20)),
    "cyclist":    dict(model=0, size=(1.80, 0.70), lr=0.60, vmax=6.0,  max_steer=0.45, participant="bicycle",     goal_dist=60.0, color=(24, 104, 225)),
    "pedestrian": dict(model=1, size=(0.70, 0.70), lr=0.50, vmax=2.0,  max_steer=1.00, participant="pedestrian",  goal_dist=20.0, color=(173, 127, 168)),
}
# Parked cars are rendered as a fifth category so they can be picked out in a video, but
# the policy only ever sees the four TYPES: in the observation a parked car is a vehicle
# at rest, indistinguishable from one stopped in traffic. Telling the policy which
# stationary vehicles will never move would be information a real one does not have.
RENDER_TYPES = TYPES + ["parked"]
PARKED_COLOR = (204, 0, 0)
PARKED_LATERAL = 1.0      # m offset towards the near side (left-hand traffic)

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
                 spawn_attempts=25, spawn_gap=1.5, pool_cap=120, n_parked=0,
                 w_progress=None, w_goal=None, w_offroad=None, w_collision=None,
                 rolling_goals=None, goal_dist=None,
                 w_redlight=None, w_wrongway=None, w_speeding=None, w_yield=None,
                 w_proximity=None, ttc_threshold=None, w_lane=None):
        self.w_lane = self.W_LANE if w_lane is None else w_lane
        self.w_proximity = self.W_PROXIMITY if w_proximity is None else w_proximity
        if ttc_threshold is not None:
            self.TTC_THRESHOLD = ttc_threshold
        self.w_yield = self.W_YIELD if w_yield is None else w_yield
        self.w_redlight = self.W_REDLIGHT if w_redlight is None else w_redlight
        self.w_wrongway = self.W_WRONGWAY if w_wrongway is None else w_wrongway
        self.w_speeding = self.W_SPEEDING if w_speeding is None else w_speeding
        # goal_dist is per type here (TYPE_SPEC), so the scalar override is ignored
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
        self.spawn_attempts = spawn_attempts
        self.spawn_gap = spawn_gap
        self.pool_cap = pool_cap
        self.n_parked = n_parked
        self._rng = np.random.default_rng(seed)
        mix = mix or DEFAULT_MIX

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
        # integer indices, not boolean masks: masked selection reads the count back from
        # the device, and the three per-type metrics below cost 12 syncs a step that way
        self._type_index = [ (self._type_idx_t == i).nonzero(as_tuple=True)[0]
                             for i in range(len(TYPES)) ]
        self.type_onehot = torch.zeros(num_agents, len(TYPES), device=device)
        self.type_onehot[torch.arange(num_agents, device=device), self._type_idx_t] = 1.0

        self._build_spawn_pools()
        self._build_parking_spots()
        self.reset()

    def _rule_participant_mask(self, lanelets):
        """Each type may only be matched to lane points of lanelets its lanelet2
        participant can pass, so a pedestrian on a crosswalk is not judged against the
        road running under it."""
        mask = torch.zeros(len(TYPES), len(lanelets), dtype=torch.bool, device=self.device)
        for i, t in enumerate(TYPES):
            rules = self._traffic_rules(TYPE_SPEC[t]["participant"])
            ok = {ll.id: rules.canPass(ll) for ll in set(lanelets)}
            mask[i] = torch.tensor([ok[ll.id] for ll in lanelets], device=self.device)
            if not bool(mask[i].any()):       # participant has no surface of its own
                mask[i] = True
        return mask

    def _signal_participant_mask(self, owners):
        """Each type is judged only against stop lines on lanelets it may use, so
        pedestrians answer to the crosswalk signals and vehicles to the road ones.
        A cyclist may pass both, and so obeys whichever it is approaching."""
        mask = torch.zeros(len(TYPES), len(owners), dtype=torch.bool, device=self.device)
        for i, t in enumerate(TYPES):
            rules = self._traffic_rules(TYPE_SPEC[t]["participant"])
            ok = {ll.id: rules.canPass(ll) for ll in set(owners)}
            mask[i] = torch.tensor([ok[ll.id] for ll in owners], device=self.device)
        return mask

    def _agent_types(self):
        return self._type_idx_t

    def _build_parking_spots(self):
        """Straight, non-junction **kerbside** lanelets a car may be left standing on.

        Junction lanelets are excluded - a car abandoned inside an intersection is a
        different (and much harsher) scenario than one at the kerb, and it would
        interact with the signal and right-of-way logic rather than test obstacle
        avoidance.

        Kerbside means no lane to the left, since Japan drives on the left and cars are
        left at the near kerb. Picking any straight lanelet put 6 of 16 cars in a
        running lane with traffic on both sides of them - an obstacle no real street
        has. Each spot also carries its own half-width, because parking is against the
        left bound, and lanes here run 2.4-3.6 m wide: one fixed offset either leaves
        the car mid-lane on a wide one or pushes it over the kerb on a narrow one.
        """
        rules = self._traffic_rules("vehicle")
        graph = self._graphs.get("vehicle")
        spots, dropped = [], 0
        for ll in self.lanelet_map.laneletLayer:
            if not rules.canPass(ll):
                continue
            if 'turn_direction' in ll.attributes:      # junction approach or turn
                continue
            pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
            if pts.shape[0] < 2:
                continue
            cache = polyline_cumlen(pts)
            if cache[2][-1] < 15.0:                    # too short to leave a car on
                continue
            veh_sl = self._sl_xy[~self._sl_ped]        # crosswalk lines are not approaches
            if veh_sl.shape[0]:
                d = float(torch.cdist(torch.tensor(pts, dtype=torch.float32, device=self.device),
                                      veh_sl).min())
                if d < self.INTERSECTION_RADIUS:       # still within a junction
                    continue
            # `left` is the neighbour a lane change may reach, `adjacentLeft` the one it
            # may not; either means this is not the kerbside lane.
            if graph is not None and (graph.left(ll) is not None
                                      or graph.adjacentLeft(ll) is not None):
                dropped += 1
                continue
            spots.append((pts, cache, self._half_width(ll, pts)))
        if not spots:
            # Synthetic maps (and the tests) can have no kerbside lanelet at all;
            # falling back keeps parking available there rather than silently off.
            spots = self._any_straight_spots(rules)
        self._parking_spots = spots
        self._parking_dropped = dropped

    def _any_straight_spots(self, rules):
        """Fallback pool when no kerbside lanelet exists, e.g. on a one-lanelet map."""
        spots = []
        for ll in self.lanelet_map.laneletLayer:
            if not rules.canPass(ll) or 'turn_direction' in ll.attributes:
                continue
            pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
            if pts.shape[0] < 2:
                continue
            cache = polyline_cumlen(pts)
            if cache[2][-1] >= 15.0:
                spots.append((pts, cache, self._half_width(ll, pts)))
        return spots

    @staticmethod
    def _half_width(ll, centre):
        """Median distance from the centreline to the left bound, i.e. the lane's half
        width on the kerb side. Taken as a median because the ends of a lanelet flare
        where it meets the next one.

        Distance to the bound as a *polyline*, not to its vertices: bounds here are
        stored sparsely (a straight edge is two points tens of metres apart), and the
        vertex distance then reports a half width many times the real one.
        """
        left = np.array([[p.x, p.y] for p in ll.leftBound], dtype=np.float64)
        if left.shape[0] < 2 or centre.shape[0] < 1:
            return PARKED_LATERAL
        a, b = left[:-1], left[1:]                       # [S, 2] segment ends
        ab = b - a
        denom = (ab * ab).sum(-1).clip(min=1e-9)
        t = (((centre[:, None, :] - a[None]) * ab[None]).sum(-1) / denom).clip(0.0, 1.0)
        foot = a[None] + t[..., None] * ab[None]         # [P, S, 2] closest point per segment
        d = np.linalg.norm(centre[:, None, :] - foot, axis=-1).min(axis=1)
        return float(np.median(d))

    def _place_parked(self, placed=None):
        """Re-place the parked cars each reset. They are ordinary vehicles as far as the
        observation is concerned; they simply never receive an action.

        `placed` is the (x, y, radius) list the spawn sampler rejected against. Parked
        cars used to be written in afterwards without consulting it, so they could be
        dropped straight on top of an agent that had just been placed clear of everyone:
        4 of 16 spent every step of the episode in contact, which is initial overlap,
        not a collision the policy caused.
        """
        if not self.n_parked or not self._parking_spots:
            return
        placed = placed if placed is not None else []
        half_l, half_w = (0.5 * TYPE_SPEC["vehicle"]["size"][0],
                          0.5 * TYPE_SPEC["vehicle"]["size"][1])
        radius = half_l + self.spawn_gap
        veh = [i for i in range(self.num_agents)
               if TYPES[self.agent_type_idx[i]] == "vehicle"][:self.n_parked]
        for i in veh:
            attempt = None
            for _ in range(self.spawn_attempts):
                poly, cache, lane_half_w = self._parking_spots[
                    self._rng.integers(len(self._parking_spots))]
                total = cache[2][-1]
                x, y, h = point_at_arclen(poly, self._rng.uniform(5.0, total - 5.0), cache)
                # against the left bound: the car's near side touches the kerb rather
                # than sitting a fixed 1 m off the lane centre
                lateral = max(lane_half_w - half_w, 0.0)
                x -= lateral * np.sin(h)
                y += lateral * np.cos(h)
                attempt = (x, y, h)
                if all((x - px) ** 2 + (y - py) ** 2 > (radius + pr) ** 2
                       for px, py, pr in placed):
                    break
            x, y, h = attempt
            placed.append((x, y, radius))
            self._init_state[0, i, 0] = x
            self._init_state[0, i, 1] = y
            self._init_state[0, i, 2] = h
            self._init_state[0, i, 3] = 0.0
            self._parked[i] = True

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

    PED_LINK_RADIUS = 12.0    # m between the end of one crossing and the start of the next
    PED_POOL_LENGTH = 80.0    # m of route to chain together before an episode starts

    def _build_ped_links(self):
        """Walkable segments and a geometric successor lookup for pedestrians.

        The map has no footway network. Of the 92 lanelets a pedestrian may use, 84 are
        crosswalks and 8 are walkways, and the routing graph gives *every one of them*
        zero successors - they are islands crossing a road, not a path along it. So a
        pedestrian walked a median 18.7 m, ran out of route and was teleported: 1260 of
        the 1550 respawns in an episode were pedestrians, and their "goals" counted
        those hops rather than 20 m walks.

        A pedestrian at a junction carries on round the corner onto the next crossing,
        so ends that nearly touch are linked here. Synthetic, like the signal phases,
        and for the same reason: the map carries the geometry but not the network.
        """
        rules = self._traffic_rules("pedestrian")
        self._ped_polys = []
        for ll in sorted((l for l in self.lanelet_map.laneletLayer if rules.canPass(l)),
                         key=lambda l: l.id):
            pts = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
            if pts.shape[0] >= 2:
                self._ped_polys.append((ll, pts))
        self._ped_ids = {ll.id for ll, _ in self._ped_polys}
        self._ped_ends = (np.array([[p[0], p[-1]] for _, p in self._ped_polys])
                          if self._ped_polys else np.zeros((0, 2, 2)))
        # validated once, because the check is a lanelet2 point query and the routes are
        # extended inside the step loop
        self._ped_succ = {(k, e): self._ped_candidates(k, pts[0 if e == 0 else -1])
                          for k, (_, pts) in enumerate(self._ped_polys) for e in (0, 1)}

    def _ped_candidates(self, k, tip):
        """Segments reachable on foot from one end of segment `k`, nearest first."""
        d0 = np.linalg.norm(self._ped_ends[:, 0] - tip, axis=1)
        d1 = np.linalg.norm(self._ped_ends[:, 1] - tip, axis=1)
        d = np.minimum(d0, d1)
        out = []
        for j in np.argsort(d):
            if d[j] > self.PED_LINK_RADIUS:
                break
            if j == k:
                continue
            ll, pts = self._ped_polys[j]
            poly = pts if d0[j] <= d1[j] else pts[::-1]
            if self._walkable_gap(tip, poly[0]):
                out.append((ll, poly))
        return out

    def _walkable_gap(self, a, b):
        """Whether the straight hop from `a` to `b` stays on ground a pedestrian may use.

        Without this the link is drawn between two crossings across whatever lies
        between them, and the map has no lanelet there: pedestrians following the route
        the env itself gave them read as off-road on 4.4% of their steps and were
        charged `w_offroad` for it. Nothing else in the run produced pedestrian offroad
        at all - it was 0.000 before the links existed.
        """
        seg = b - a
        n = max(int(np.linalg.norm(seg) / 1.5), 1)
        for f in np.linspace(0.0, 1.0, n + 1):
            x, y = a + f * seg
            p = lanelet2.core.BasicPoint2d(float(x), float(y))
            near = lanelet2.geometry.findNearest(self.lanelet_map.laneletLayer, p, 6)
            if not any(dist <= 0.01 and ll.id in self._ped_ids for dist, ll in near):
                return False
        return True

    def _ped_next(self, xy, exclude_id):
        """The next walkable segment leading away from `xy`, oriented to continue from
        it, or None if nothing starts close enough. Returns (lanelet, polyline)."""
        if not self._ped_polys:
            return None
        d = np.minimum(np.linalg.norm(self._ped_ends[:, 0] - xy, axis=1),
                       np.linalg.norm(self._ped_ends[:, 1] - xy, axis=1))
        k = int(np.argmin(d))
        e = 0 if np.linalg.norm(self._ped_ends[k, 0] - xy) <= np.linalg.norm(
            self._ped_ends[k, 1] - xy) else 1
        for ll, poly in self._ped_succ.get((k, e), ()):
            if ll.id != exclude_id:
                return ll, poly
        return None

    def _chain_ped_route(self, ll, pts):
        """Link crossings end to end until the route is worth walking."""
        cache = polyline_cumlen(pts)
        end = ll
        while cache[2][-1] < self.PED_POOL_LENGTH:
            nxt = self._ped_next(pts[-1], end.id)
            if nxt is None:
                break
            end, ext = nxt
            if np.allclose(pts[-1], ext[0], atol=1e-6):
                ext = ext[1:]
            if ext.shape[0] < 1:
                break
            pts = np.concatenate([pts, ext], axis=0)
            cache = polyline_cumlen(pts)
        return pts, end

    def _build_spawn_pools(self):
        """One pool of (polyline, arc-length cache) per participant, built once
        from that participant's traffic rules (canPass) and routing graph."""
        participants = {TYPE_SPEC[t]["participant"] for t in TYPES}
        self._build_ped_links()
        self._pools, self._graphs = {}, {}
        for pkey in participants:
            rules = self._traffic_rules(pkey)
            graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
            self._graphs[pkey] = graph
            polys = []
            for ll in self._passable_lanelets(rules):
                if pkey == "pedestrian":  # chained geometrically; see _build_ped_links
                    poly, end = self._chain_ped_route(
                        ll, np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64))
                else:                      # wheeled agents follow a legal downstream route
                    poly, end = build_route(graph, ll, return_end=True)
                if poly.shape[0] >= 2:
                    cache = polyline_cumlen(poly)
                    if cache[2][-1] > 3.0:
                        # `end` is where the route stopped; rolling goals continue from it
                        polys.append((poly, cache, end))
            self._pools[pkey] = polys
        for pkey in participants:  # any empty pool falls back to the vehicle network
            if not self._pools[pkey]:
                self._pools[pkey] = self._pools.get("vehicle", [])

    def _sample_one(self, spec, placed):
        """Pick a collision-free start along a route from this type's pool. The goal sits
        `goal_dist` ahead and is rolled forward on arrival (AWSIMDrivingEnv._advance_goals)."""
        pool = self._pools[spec["participant"]]
        radius = 0.5 * max(spec["size"]) + self.spawn_gap
        best, best_clear = None, -np.inf
        for _ in range(self.spawn_attempts):
            poly, cache, end = pool[self._rng.integers(len(pool))]
            total = cache[2][-1]
            s0 = self._rng.uniform(0.0, max(total - 3.0, 0.0) * 0.6)
            x, y, h = point_at_arclen(poly, s0, cache)
            gx, gy, _ = point_at_arclen(poly, min(s0 + spec["goal_dist"], total), cache)
            attempt = (x, y, h, gx, gy, poly, end, s0)
            clear = min((np.hypot(x - px, y - py) - (radius + pr) for px, py, pr in placed),
                        default=np.inf)
            if clear > 0:
                return attempt
            # Every attempt overlapping used to mean "take the last one", which put 10
            # pedestrians into the episode already in contact - their pool is 92 short
            # lanelets for 153 of them, so overlap is the normal case, not the rare one.
            # Keeping the roomiest attempt costs nothing and starts them apart.
            if clear > best_clear:
                best, best_clear = attempt, clear
        return best

    def _respawn(self, i):
        """Put agent i on a fresh route once its own runs out at the edge of the map,
        keeping it clear of where the other agents currently are."""
        spec = TYPE_SPEC[TYPES[self.agent_type_idx[i]]]
        # positions are pulled to the host once per batch of respawns, not once per
        # agent: this used to be a full 512-agent device read inside the loop
        xy = self._respawn_xy
        # `spawn_gap` as well as the body, matching _sample_spawns: with a bare 3 m
        # radius, 7.2% of respawns still landed touching somebody
        placed = [(px, py, 0.5 * max(spec["size"]) + self.spawn_gap)
                  for j, (px, py) in enumerate(xy) if j != i]
        x, y, h, _, _, poly, end, s0 = self._sample_one(spec, placed)
        self._route_poly[i] = poly
        self._route_cache[i] = polyline_cumlen(poly)
        self._route_end_ll[i] = end
        self._route_pkey[i] = spec["participant"]
        self._s_goal[i] = s0
        return x, y, h

    def _sample_spawns(self):
        placed = []  # (x, y, radius)
        starts, headings, goals = [], [], []
        polys, ends, pkeys, s0s, gaps = [], [], [], [], []
        for tidx in self.agent_type_idx:
            spec = TYPE_SPEC[TYPES[tidx]]
            x, y, h, gx, gy, poly, end, s0 = self._sample_one(spec, placed)
            placed.append((x, y, 0.5 * max(spec["size"]) + self.spawn_gap))
            starts.append((x, y)); headings.append(h); goals.append((gx, gy))
            polys.append(poly); ends.append(end); pkeys.append(spec["participant"])
            s0s.append(s0); gaps.append(spec["goal_dist"])

        A = self.num_agents
        self._parked = torch.zeros(A, dtype=torch.bool, device=self.device)
        self._route_end_ll, self._route_pkey = ends, pkeys
        self._init_route_state(polys, s0s, gaps)
        self._init_state = torch.zeros(1, A, 4, device=self.device)
        self._init_state[0, :, :2] = torch.tensor(np.stack(starts), dtype=torch.float32, device=self.device)
        self._init_state[0, :, 2] = torch.tensor(np.stack(headings), dtype=torch.float32, device=self.device)
        self._place_parked(placed)

    def _extend_route(self, i):
        """Continue agent i's route through the lanelet graph so its goals can keep
        rolling forward. False at a dead end, which parks that agent."""
        if self._route_pkey[i] == "pedestrian":
            nxt = self._ped_next(self._route_poly[i][-1], self._route_end_ll[i].id)
            if nxt is None:
                return False
            end, ext = nxt
            return self._append_route(i, ext, end)
        graph = self._graphs[self._route_pkey[i]]
        nxt = list(graph.following(self._route_end_ll[i]))
        if not nxt:
            return False
        ext, end = build_route(graph, nxt[0], return_end=True)
        if ext.shape[0] < 2:
            return False
        return self._append_route(i, ext, end)

    def _append_route(self, i, ext, end):
        """Tack `ext` onto agent i's route, now ending on lanelet `end`."""
        poly = self._route_poly[i]
        if np.allclose(poly[-1], ext[0], atol=1e-6):   # drop the duplicated junction point
            ext = ext[1:]
        if ext.shape[0] < 1:
            return False
        # concatenating makes a fresh array, so the pool's shared polyline is untouched
        self._route_poly[i] = np.concatenate([poly, ext], axis=0)
        self._route_cache[i] = polyline_cumlen(self._route_poly[i])
        self._route_end_ll[i] = end
        return True

    def _limit_steering(self, action, state):
        """The bicycle limits, plus the pedestrian's own speed cap.

        `OrientedKinematicModel` normalises x and y independently, so a pedestrian
        walking diagonally covered vmax * sqrt(2) = 2.83 m/s against its 2.0 m/s cap -
        and the measured median step was exactly that. `_post_physics` clamps the state's
        speed component, which the omnidirectional model does not use to move, so it
        never bit. Capping the norm of the body-frame step does.
        """
        action = super()._limit_steering(action, state)
        walker = ~self._steer_wheeled
        if bool(walker.any()):
            norm = action[:, :2].norm(dim=-1, keepdim=True).clamp(min=1.0)
            action[:, :2] = torch.where(walker.unsqueeze(1), action[:, :2] / norm,
                                        action[:, :2])
        return action

    def _speed_limit(self, near):
        """No type may be judged against a limit it cannot reach. A cyclist was measured
        against the 50 km/h of the road it rides on while its own top speed is 21.6, so
        a speeding violation was arithmetically impossible and its speed/limit ratio was
        not comparable with any other type's."""
        return torch.minimum(super()._speed_limit(near), self.vmax)

    def _steering_params(self):
        """Per-agent rear-axle distance, steering limit, and which agents are on the
        bicycle model at all - the pedestrian model's second action is a sideways step,
        not a steering angle, so `_limit_steering` must leave it alone."""
        spec = [TYPE_SPEC[TYPES[t]] for t in self.agent_type_idx]
        lr = torch.tensor([s["lr"] for s in spec], dtype=torch.float32, device=self.device)
        mx = torch.tensor([s["max_steer"] for s in spec], dtype=torch.float32, device=self.device)
        wheeled = torch.tensor([s["model"] == 0 for s in spec], device=self.device)
        return lr, mx, wheeled

    def _build_simulator(self):
        A = self.num_agents
        assign = self.model_assignments
        wheeled, ped = (assign[0] == 0), (assign[0] == 1)
        bike = SteeringLimitedBicycle(dt=self.dt)
        bike.set_params(lr=self.lr_all[wheeled].clone())
        bike.set_steering_limit(self._steering_params()[1][wheeled].clone())
        bike.set_state(self._init_state[0, wheeled].clone())
        # OrientedKinematicModel, not SimpleKinematicModel: the latter's action is the
        # world-frame state gradient, but observations are purely ego-frame (the agent's
        # own psi is not observable), so a pedestrian cannot work out which world
        # direction its goal lies in. The oriented model rotates the action frame with the
        # agent, matching the bicycle agents' body-frame actions.
        walk = OrientedKinematicModel(dt=self.dt, max_dx=TYPE_SPEC["pedestrian"]["vmax"])
        walk.set_state(self._init_state[0, ped].clone())
        kin = CompoundKinematicModel([bike, walk], model_assignments=assign, dt=self.dt)
        # State/params are already on device, but each model's action `_normalization_factor`
        # is built on CPU in the constructor - move the whole model over.
        kin = kin.to(self.device)

        renderer = renderer_from_config(RendererConfig(left_handed_coordinates=False))
        for t in TYPES:
            renderer.color_map[t] = TYPE_SPEC[t]["color"]
            renderer.rendering_levels.setdefault(t, 4)
        renderer.color_map["parked"] = PARKED_COLOR
        renderer.rendering_levels.setdefault("parked", 4)
        # render-only category: the observation's type one-hot still says "vehicle"
        render_idx = torch.where(self._parked, torch.full_like(self._type_idx_t, len(TYPES)),
                                 self._type_idx_t)
        cfg = TorchDriveConfig(left_handed_coordinates=False,
                               renderer=RendererConfig(left_handed_coordinates=False))
        self.simulator = Simulator(
            cfg=cfg, road_mesh=self.mesh, kinematic_model=kin, agent_size=self.agent_size,
            initial_present_mask=torch.ones(1, A, dtype=torch.bool, device=self.device),
            renderer=renderer, lanelet_map=[self.lanelet_map],
            agent_types=render_idx.view(1, A), agent_type_names=RENDER_TYPES,
        )

    # ---------------------------------------------------- overridden hooks
    def _prepare_reset(self):
        self._sample_spawns()  # re-randomise spawns every episode

    def _ego_features(self, state, prev_action):  # append own type one-hot
        return torch.cat([super()._ego_features(state, prev_action), self.type_onehot], dim=-1)

    def _partner_extra(self, nidx, valid):        # append each neighbour's type one-hot
        return self.type_onehot[nidx]

    def _post_physics(self):
        # writing back unconditionally: the torch.equal guard that used to be here read
        # the device every step purely to decide whether to skip a write that costs less
        # than the synchronisation did
        state = self._state()
        new_state = state.clone()
        new_state[:, 3] = torch.clamp(state[:, 3], -self.vmax, self.vmax)  # per-type cap
        self.simulator.set_state(new_state.unsqueeze(0))

    def _augment_info(self, info):
        sr = self._speed_ratio(self._state())
        for i, t in enumerate(TYPES):  # per-type goal-reaching
            m = self._type_index[i]
            info[f'reached_{t}'] = self._reached[m].float().mean()
            info[f'goals_{t}'] = self._goals_reached[m].mean()
            # per type, because the mixed average hides which types have stopped: when
            # the wheeled agents parked, the overall ratio was 0.22 while pedestrians
            # were still at 0.53 of their limit
            info[f'speed_{t}'] = sr[m].mean()
