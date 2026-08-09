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


class SteeringLimitedBicycle(KinematicBicycle):
    """`KinematicBicycle` whose steering limit may differ per agent.

    The library normalises steering by one scalar `max_steering`, so a mixed scene has
    to give a bicycle and a lorry the same slip angle for the same action. Only the
    normalisation is per-agent here; the dynamics are the library's unchanged.
    """

    def set_steering_limit(self, max_steering):
        """`max_steering`: [A] tensor of radians, one per agent of this model."""
        accel = torch.full_like(max_steering, float(self.max_acceleration))
        self._normalization_factor = torch.stack([accel, max_steering], dim=-1)


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
    EGO_DIM = 15              # speed, goal, prev action, signal (5), limit, must-yield
    BRANCH_DIM = 27           # 3 branches x (4 preview points + a validity bit)
    # Where each lane the agent could legally be in goes next, rather than a bearing to
    # one point far along one of them. Three branches - the lane it is on and the two it
    # may change into - each described by points at fixed distances ahead, plus a bit
    # saying the branch exists. This is an affordance, not an instruction: every branch
    # is equally rewarded, so *when* to change lane stays a decision the policy makes.
    BRANCH_SPACING = 2.0             # m between points of the resampled lane path
    BRANCH_POINTS = 31               # so each path covers 60 m
    BRANCH_PREVIEW = (3, 5, 10, 17)  # indices ahead, i.e. 6, 10, 20 and 34 m
    BRANCH_HALF_WIDTH = 2.0          # m of lateral error at which the follow score is 0
    W_FOLLOW = 0.0                   # replaces w_progress for wheeled agents when > 0
    # Off by default: turning it on changes OBS_DIM, so every run in a comparison has to
    # agree about it. `w_follow > 0` implies it.
    BRANCH_OBS = False
    # What happens when a route runs out of map. The default puts the agent on a fresh
    # route somewhere else, which is why `reached` reads 0.00 in every log: a route is
    # effectively infinite and there is no such thing as arriving. Despawning instead
    # makes the last goal final - the agent stops being simulated, stops colliding and
    # stops being drawn - at the cost of the scene thinning out over an episode.
    DESPAWN_AT_GOAL = False
    # A respawn moves the agent to an unrelated part of the map, but `done` was not set
    # for it, so GAE bootstrapped V(somewhere else) into the step that earned the goal -
    # the agent was scored on wherever it happened to be dropped. Terminating the
    # transition makes the last goal a real episode boundary for that agent, after which
    # it starts again from a freshly drawn spawn point.
    TERMINATE_ON_TELEPORT = False
    # `_offroad` returns metres - the summed distance of the four corners past the edge -
    # not an indicator, and two places treated it as one. `progress * (1 - offroad)` was
    # meant to withhold credit off-road; with a magnitude it *reverses* the sign, and
    # measured on a trained policy 72% of off-road steps came out with a negative
    # multiplier, median -1.07 and mean -9.5. An agent off the road was paid to stop, and
    # paid to stay off, which is why excursions ran long enough to trip the lost counter.
    # The penalty side was unbounded for the same reason: 0.6 x a median 2.07 is 1.24 a
    # step against a forward incentive of 0.095, and the 90th percentile is 25.
    OFFROAD_FIX = False
    OFFROAD_CAP = 1.0                # metres of overhang past which it costs no more
    # The crossing risk put both agents at their centres with a radius of their half
    # width, so a car's 4.97 m of length was invisible to it. See _proximity.
    CAPSULE_RISK = False
    # The proximity term took the worst partner over all of them, so a pedestrian
    # stepping off a kerb was invisible behind the car in front: 48% of a vehicle's
    # partners classify as crossing and 30% of those are pedestrians, but none of it
    # reaches the reward while a same-lane leader scores higher. One worst partner *per
    # type*, summed, keeps a pedestrian's risk from being hidden by a car's.
    PROXIMITY_PER_TYPE = False
    # `failtoyield` cannot fire for a pedestrian: the map's right_of_way elements only
    # relate road lanelets, and the crosswalk conflict was left unmodelled on purpose.
    # A crossing that a road lanelet passes through gives that road lanelet something to
    # give way to.
    CROSSWALK_PRIORITY = False
    MAX_PARTNERS = 16         # K nearest neighbours observed
    PARTNER_FEATURES = 8      # rel_x/y, width, length, rel_head_cos/sin, rel_speed, has_priority
    # 30 m was shorter than the braking distance of the faster agents: a motorcycle at
    # its 65 km/h vmax needs 32.4 m to stop at 5 m/s^2, so 53% of its collisions were
    # already unavoidable at the moment the partner first became observable. Widening the
    # radius alone would not have helped - with 6.9 neighbours inside 30 m on average, the
    # 8 nearest slots were filled by close agents and the distant one still never appeared
    # - so MAX_PARTNERS goes up with it.
    VIEW_RADIUS = 60.0        # metres; neighbours beyond this are not observed
    MAX_ROAD = 10             # forward distance bands of road-graph points observed
    ROAD_FEATURES = 4         # rel_x, rel_y, rel_dir_cos, rel_dir_sin
    ROAD_RADIUS = 50.0        # metres of road ahead described, one point per band
    ROAD_BEHIND = 5.0         # the first band starts here, so the point underfoot is seen
    # Banding needs a cloud finer than the bands: at the old 2000 points the spacing was
    # ~21 m over 41.7 km of lane, so most 5.5 m bands would have been empty.
    ROAD_POINT_CAP = 8000     # subsample the lane-centreline point cloud to this many
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
    # Leaving the road has to cost more than the shortcut it buys: driving straight at
    # the goal through a building earns w_progress * v * dt = 0.14 per step at 14 m/s,
    # against which an 0.1 penalty was a net profit of 0.04 - and the policy took it.
    # Measured at w_offroad=0.1, excursions ran to 214 steps (21 s, a whole episode) and
    # runs of 3 s or longer accounted for 71% of all off-road time.
    W_OFFROAD = 0.6
    # Progress is not credited while off the drivable surface, so cutting a corner earns
    # nothing rather than merely costing a little.
    # An agent that has been off-road this long is not clipping a kerb, it is lost, and
    # is put back on a fresh route rather than left to drive through buildings for the
    # rest of the episode (the median excursion is 9 steps, the 90th percentile 55).
    OFFROAD_PATIENCE = 20
    # Collision has to outweigh the progress given up by avoiding one. Slowing from v to
    # a stop costs w_progress * v * dt per step - 0.14 at 14 m/s with w_progress=0.1 -
    # while the collision itself only costs W_COLLISION per step, so at 0.1 driving
    # through was strictly cheaper than yielding and the collision rate never moved
    # across four runs. Same shape as the speeding weight; see W_SPEEDING.
    W_COLLISION = 0.6
    # Contact-only penalties give no gradient before contact: relu(1 - d/(r_i+r_j)) is
    # exactly zero while the agents are apart, so "too close" costs nothing. Measured,
    # 65% of collisions were still avoidable by braking when the partner first became
    # visible, and the mean throttle from first sight to impact was +1.2 - the policy
    # accelerates into agents it can see. This penalises closing time instead, so the
    # signal exists before the boxes overlap.
    W_PROXIMITY = 0.3
    TTC_THRESHOLD = 3.0              # seconds; contact predicted sooner than this costs
    PROXIMITY_MARGIN = 0.5           # metres of lateral clearance counted as a conflict
    RSS_REACTION = 0.3               # s of response time before the ego can brake
    RSS_ACCEL = 2.0                  # m/s^2 assumed during that response
    RSS_BRAKE = 5.0                  # m/s^2 braking, matching KinematicBicycle
    CORRIDOR_HALF_WIDTH = 6.0        # m; a loose sanity bound, not the lane test itself
    DOWNSTREAM_HOPS = 3              # lanelets ahead still counted as the same road

    # Steering limits. `KinematicBicycle` normalises steering by a single `max_steering`
    # that defaults to pi/2, so one unit of action was a *90 degree* slip angle for every
    # road user, and the heading rate `psi_dot = v / lr * sin(beta)` then scales with
    # 1 / lr - the shortest wheelbase spins fastest for the same command. Measured on the
    # trained policy, motorcycles (lr = 0.8 m) averaged 515 deg/s of heading change, 51
    # degrees per step, with the steering pinned to the rail 8.7% of the time: a top, not
    # a vehicle, and it still scored second best on goals. Three limits, in order:
    #   1. MAX_STEER    - the geometric limit the action range is normalised to.
    #   2. LAT_ACCEL_MAX - sin(beta) <= a_lat * lr / v^2, since a_lat = v^2 / lr * sin(beta).
    #      This is what stops the spinning: it is speed-dependent, so a tight turn is
    #      only available after slowing down, exactly as on a real road.
    #   3. STEER_RATE_MAX - how fast the steering itself may move between steps.
    MAX_STEER = 0.30                 # rad (17 deg) of slip angle at the geometric centre
    LAT_ACCEL_MAX = 4.0              # m/s^2 of lateral acceleration
    STEER_RATE_MAX = 2.0             # rad/s, i.e. 0.2 rad of change per 0.1 s step

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
    # Speeding: free up to SPEED_TOLERANCE over the limit, then quadratic in the excess
    # beyond it. A few km/h over is tolerated in practice and a lot is not; a plain
    # linear term cannot say both, and the earlier ratio form was scale-free, which made
    # a 10 km/h zone proportionally as forgiving as a 50 km/h one - backwards.
    #
    #   penalty = min(W_SPEEDING * over^2, SPEED_PENALTY_CAP),  over = max(0, v - limit - tol)
    #
    # The cap is a safety rail rather than a shaping choice: a motorcycle at its 65 km/h
    # vmax in one of the map's 10 km/h zones is 15 m/s over, and 0.05 * 12.4^2 is 7.7 in
    # a single step, enough to distort the value function on its own.
    W_SPEEDING = 0.05
    SPEED_TOLERANCE = 10 / 3.6       # m/s over the limit that costs nothing
    SPEED_PENALTY_CAP = 4.0          # per step; only binds past ~+43 km/h
    # Failure to yield: the map's right_of_way elements say which lanelets must give way
    # to which. Note what is NOT used here - the other agent's intention. A real vehicle
    # knows the map and can localise its neighbours, so "that agent is on a lanelet with
    # priority over mine" is fair game; "that agent is about to go" is not, and a policy
    # trained on it would not transfer off the simulator.
    W_YIELD = 0.5
    YIELD_RADIUS = 25.0              # metres; a priority agent nearer than this must be given way
    YIELD_SPEED = 1.0                # m/s; below this the agent counts as having stopped
    # Right turns crossing oncoming traffic are the conflict the map does not describe
    # and the signal phases cannot separate - see _add_oncoming_turn_priority.
    TURN_CONFLICT_DIST = 3.0         # metres; how close two centrelines must come to conflict
    TURN_CONFLICT_RADIUS = 60.0      # metres; only consider lanelets this near each other
    # vehicle vmax (14.0 m/s) sits a hair above the 50 km/h limit (13.89), so counting
    # any excess at all reports a full-speed car on a main road as a violation. Only
    # count a real margin, and report the magnitude separately.
    SPEEDING_TOLERANCE = 0.05

    # Lane keeping. `offroad` only fires once the body has already left the drivable
    # surface, which is a late and sparse signal: 73% of the off-road steps measured on
    # the trained policy happened on straight lanelets, not at the junctions where
    # cornering is hard, so agents were drifting out of lane with nothing pushing back
    # until they were already out. This charges lateral offset from the lane centre
    # before that, and it is a penalty rather than a bonus on purpose - a bonus for
    # sitting on a centreline is collected just as well by an agent that has stopped.
    # The rise alone makes staying overlapped free once you are in, and the overlap
    # time went up 18% because of it. A small level term restores a reason to get out.
    # Changing lane was free, and stepping sideways is the cheapest way out of a
    # rear-end conflict. `w_lanechange` charges a legal change (a dashed line, which is
    # normal driving and so should cost little); `w_solidcross` charges crossing a line
    # the map says may not be crossed.
    W_LANECHANGE = 0.0
    W_SOLIDCROSS = 0.0
    W_COLLISION_LEVEL = 0.0
    W_LANE = 0.4
    LANE_TOLERANCE = 1.2             # m of lateral offset that costs nothing
    LANE_SCALE = 1.5                 # m over which the penalty ramps to its full value

    def __init__(self, map_path, num_agents=8, max_steps=80, dt=0.1, device='cpu',
                 goal_radius=3.0, render_fov=None, render_res=512, seed=0,
                 w_progress=None, w_goal=None, w_offroad=None, w_collision=None,
                 rolling_goals=None, goal_dist=None,
                 w_redlight=None, w_wrongway=None, w_speeding=None, w_yield=None,
                 w_proximity=None, ttc_threshold=None, w_lane=None,
                 w_collision_level=None, lat_accel_max=None,
                 w_lanechange=None, w_solidcross=None, max_steer_scale=1.0,
                 w_follow=None, branch_obs=None, despawn_at_goal=None,
                 terminate_on_teleport=None, offroad_fix=None, capsule_risk=None,
                 proximity_per_type=None, crosswalk_priority=None):
        self.capsule_risk = self.CAPSULE_RISK if capsule_risk is None else capsule_risk
        self.proximity_per_type = (self.PROXIMITY_PER_TYPE if proximity_per_type is None
                                   else proximity_per_type)
        self.crosswalk_priority = (self.CROSSWALK_PRIORITY if crosswalk_priority is None
                                   else crosswalk_priority)
        self.offroad_fix = self.OFFROAD_FIX if offroad_fix is None else offroad_fix
        self.terminate_on_teleport = (self.TERMINATE_ON_TELEPORT
                                      if terminate_on_teleport is None
                                      else terminate_on_teleport)
        self.despawn_at_goal = (self.DESPAWN_AT_GOAL if despawn_at_goal is None
                                else despawn_at_goal)
        self.w_follow = self.W_FOLLOW if w_follow is None else w_follow
        self.branch_obs = ((self.BRANCH_OBS if branch_obs is None else branch_obs)
                           or self.w_follow > 0)
        if self.branch_obs:       # instance attribute, so the trainer sizes the net right
            self.EGO_DIM = type(self).EGO_DIM + self.BRANCH_DIM
            self.OBS_DIM = (self.EGO_DIM + self.MAX_PARTNERS * self.PARTNER_FEATURES
                            + self.MAX_ROAD * self.ROAD_FEATURES)
        self.max_steer_scale = max_steer_scale
        self.w_lanechange = self.W_LANECHANGE if w_lanechange is None else w_lanechange
        self.w_solidcross = self.W_SOLIDCROSS if w_solidcross is None else w_solidcross
        self.w_collision_level = (self.W_COLLISION_LEVEL if w_collision_level is None
                                  else w_collision_level)
        if lat_accel_max is not None:
            self.LAT_ACCEL_MAX = lat_accel_max
        self.w_lane = self.W_LANE if w_lane is None else w_lane
        self.w_proximity = self.W_PROXIMITY if w_proximity is None else w_proximity
        if ttc_threshold is not None:
            self.TTC_THRESHOLD = ttc_threshold
        self.w_yield = self.W_YIELD if w_yield is None else w_yield
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
        self._build_right_of_way(lls)

    def _build_right_of_way(self, lanelets):
        """Which lanelet must give way to which, from the map's right_of_way elements.

        `_priority[i, j]` is True when lanelet j has right of way over lanelet i. The
        matrix is dense (L x L bools, ~1 MB at 979 lanelets) so the per-step lookup is a
        single gather.
        """
        ids = sorted({ll.id for ll in lanelets})
        order = {i: k for k, i in enumerate(ids)}
        L = len(ids)
        priority = torch.zeros(L, L, dtype=torch.bool, device=self.device)
        yields = torch.zeros(L, dtype=torch.bool, device=self.device)
        seen, parsed = set(), 0
        for ll in self.lanelet_map.laneletLayer:
            for r in ll.regulatoryElements:
                if 'subtype' not in r.attributes or r.attributes['subtype'] != 'right_of_way':
                    continue
                if r.id in seen:
                    continue
                seen.add(r.id)
                # these are methods, not properties - reading them as attributes
                # yields a bound method and silently produces empty tables
                prio = [x.id for x in r.rightOfWayLanelets()]
                give = [x.id for x in r.yieldLanelets()]
                parsed += 1
                for y in give:
                    if y not in order:
                        continue
                    yields[order[y]] = True
                    for pz in prio:
                        if pz in order:
                            priority[order[y], order[pz]] = True
        if seen and not parsed:
            raise RuntimeError('right_of_way elements found but none parsed')
        if self.crosswalk_priority:
            parsed += self._add_crosswalk_priority(order, priority, yields)
        self._lanelet_order = order
        self._priority = priority
        self._yield_lanelet = yields
        self._pt_lanelet = torch.tensor([order[ll.id] for ll in lanelets], device=self.device)
        self._add_oncoming_turn_priority()
        self._build_downstream()
        self._build_lateral()
        self._build_branch_paths()

    def _build_downstream(self):
        """[L, L] mask: lanelet j follows lanelet i within DOWNSTREAM_HOPS.

        Used to decide whether a vehicle is 'ahead on the same road' without any
        geometric assumption about how straight that road is.
        """
        order = self._lanelet_order
        L = len(order)
        adj = torch.zeros(L, L, dtype=torch.bool, device=self.device)
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
        for ll in self.lanelet_map.laneletLayer:
            if ll.id not in order:
                continue
            for nxt in graph.following(ll):
                if nxt.id in order:
                    adj[order[ll.id], order[nxt.id]] = True
        reach = adj.clone()
        for _ in range(self.DOWNSTREAM_HOPS - 1):     # transitive closure, bounded
            reach = reach | (reach.float() @ adj.float() > 0)
        self._downstream = reach

    def _build_lateral(self):
        """[L, L] masks for stepping sideways onto a neighbouring lane.

        Nothing charged a lane change before. `offroad` fires only off the drivable
        surface, and both lanes are on it; `wrongway` needs more than 90 degrees of
        heading error; and `_lane_offset` measures to the nearest centreline *of any
        lane the participant may use*, so once the agent is over the line the offset
        resets to zero. The cheapest way out of a rear-end conflict was therefore to
        step sideways, and it was free - which is a plausible reading of why crossing
        conflicts went from 25.5% to 39.7% of contacts once driving into one got
        expensive.

        lanelet2 already draws the distinction the map does: `left`/`right` is a
        neighbour a lane change may legally reach (a dashed line - 285 of the 884
        drivable lanelets have one), `adjacentLeft`/`adjacentRight` is a neighbour it
        may not (a solid line - 201 more). The map backs this up with 486 solid and
        254 dashed boundary linestrings.
        """
        order = self._lanelet_order
        L = len(order)
        legal = torch.zeros(L, L, dtype=torch.bool, device=self.device)
        solid = torch.zeros(L, L, dtype=torch.bool, device=self.device)
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
        for ll in self.lanelet_map.laneletLayer:
            if ll.id not in order:
                continue
            i = order[ll.id]
            for nb, table in ((graph.left(ll), legal), (graph.right(ll), legal),
                              (graph.adjacentLeft(ll), solid),
                              (graph.adjacentRight(ll), solid)):
                if nb is not None and nb.id in order:
                    table[i, order[nb.id]] = True
        self._lane_legal, self._lane_solid = legal, solid & ~legal

    def _build_branch_paths(self):
        """For every drivable lanelet, the road ahead of it as a resampled polyline, and
        the indices of the lanes to its left and right.

        Built once. The successor chain takes the first `following` at each step, which
        is arbitrary at a junction, but the block is a description of the road within
        60 m and the junction geometry itself carries most of that.
        """
        order = self._lanelet_order
        L, P, sp = len(order), self.BRANCH_POINTS, self.BRANCH_SPACING
        xy = np.zeros((L, P, 2), dtype=np.float64)
        hd = np.zeros((L, P), dtype=np.float64)
        left = np.full(L, -1, dtype=np.int64)
        right = np.full(L, -1, dtype=np.int64)
        rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, rules)
        by_id = {ll.id: ll for ll in self.lanelet_map.laneletLayer if ll.id in order}
        for lid, i in order.items():
            ll = by_id.get(lid)
            if ll is None:
                continue
            for nb, table in ((graph.left(ll), left), (graph.right(ll), right)):
                if nb is not None and nb.id in order:
                    table[i] = order[nb.id]
            pts, seen, cur, total = [], {ll.id}, ll, 0.0
            while total < P * sp:
                pts.extend([(q.x, q.y) for q in cur.centerline])
                total += float(lanelet2.geometry.length(cur.centerline))
                nxt = [n for n in graph.following(cur) if n.id not in seen]
                if not nxt:
                    break
                cur = nxt[0]
                seen.add(cur.id)
            a = np.asarray(pts, dtype=np.float64)
            if a.shape[0] < 2:
                a = np.repeat(np.asarray([[ll.centerline[0].x, ll.centerline[0].y]]), 2, 0)
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
            want = np.minimum(np.arange(P) * sp, cum[-1])
            xy[i, :, 0] = np.interp(want, cum, a[:, 0])
            xy[i, :, 1] = np.interp(want, cum, a[:, 1])
            d = np.diff(xy[i], axis=0)
            hd[i, :-1] = np.arctan2(d[:, 1], d[:, 0])
            hd[i, -1] = hd[i, -2]
        self._branch_xy = torch.tensor(xy, dtype=torch.float32, device=self.device)
        self._branch_dir = torch.tensor(hd, dtype=torch.float32, device=self.device)
        self._branch_side = torch.tensor(np.stack([np.arange(L), left, right], axis=-1),
                                         device=self.device)          # [L, 3]

    def _branch_types(self):
        """Which agents get branches. One vehicle type in the base env, so all of them."""
        return torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

    def _branches_cached(self, state):
        """The observation block and the follow score both want this, and it is the
        second most expensive thing in the step after the collision matrix."""
        key = state.data_ptr(), state._version
        if getattr(self, '_br_key', None) != key:
            self._br_key, self._br_val = key, self._branches(state)
        return self._br_val

    def _branches(self, state):
        """Lateral error, heading error and preview points for the three branches.

        Returns (lat [A,3], head_err [A,3], valid [A,3], preview [A,3,K,2] in ego frame).
        """
        lane = self._agent_lanelet(state)
        idx = self._branch_side[lane]                                   # [A, 3]
        valid = (idx >= 0) & self._branch_types().unsqueeze(1)
        j = idx.clamp(min=0)
        path = self._branch_xy[j]                                       # [A, 3, P, 2]
        # closest point on the polyline, by segment so the 2 m spacing does not add to
        # the lateral error the way it did in _lane_offset
        a, b = path[:, :, :-1, :], path[:, :, 1:, :]
        ab = b - a
        den = (ab * ab).sum(-1).clamp(min=1e-9)
        rel = state[:, None, None, :2] - a
        t = ((rel * ab).sum(-1) / den).clamp(0.0, 1.0)
        foot = a + t.unsqueeze(-1) * ab
        d = (state[:, None, None, :2] - foot).norm(dim=-1)              # [A, 3, P-1]
        lat, seg = d.min(dim=-1)
        head = torch.gather(self._branch_dir[j], 2, seg.unsqueeze(-1)).squeeze(-1)
        head_err = torch.atan2(torch.sin(head - state[:, 2:3]),
                               torch.cos(head - state[:, 2:3]))
        k = torch.tensor(self.BRANCH_PREVIEW, device=self.device)
        take = (seg.unsqueeze(-1) + k).clamp(max=self.BRANCH_POINTS - 1)  # [A, 3, K]
        pts = torch.gather(path, 2, take.unsqueeze(-1).expand(-1, -1, -1, 2))
        r = pts - state[:, None, None, :2]
        c, sn = torch.cos(state[:, 2]), torch.sin(state[:, 2])
        ego = torch.stack([c[:, None, None] * r[..., 0] + sn[:, None, None] * r[..., 1],
                           -sn[:, None, None] * r[..., 0] + c[:, None, None] * r[..., 1]],
                          dim=-1) / (self.BRANCH_POINTS * self.BRANCH_SPACING)
        return lat, head_err, valid, ego * valid[..., None, None]

    def _branch_block(self, state):
        """[A, 3 * (2K + 1)] - the branch preview as the policy sees it."""
        _, _, valid, ego = self._branches_cached(state)
        A = state.shape[0]
        return torch.cat([ego.reshape(A, 3, -1), valid.float().unsqueeze(-1)],
                         dim=-1).reshape(A, -1)

    def _follow_score(self, state):
        """How well the agent is tracking *any* of its branches, in [0, 1].

        A max over branches, so nothing here says which lane to be in - only that it
        should be in one of them, pointing the way that lane goes.
        """
        lat, head_err, valid, _ = self._branches_cached(state)
        score = ((1.0 - lat / self.BRANCH_HALF_WIDTH).clamp(0.0, 1.0)
                 * torch.relu(torch.cos(head_err)))
        return (score * valid).max(dim=1).values

    def _lane_change(self, state):
        """(legal, over-a-solid-line) indicators for stepping onto a neighbouring lane
        this step. A teleport moves the agent between lanelets discontinuously, so the
        step after one is exempt, as is the first step of an episode."""
        lane = self._agent_lanelet(state)
        prev = self._prev_lane
        moved = (lane != prev) & self._lane_valid & (~self._just_moved)
        legal = moved & self._lane_legal[prev, lane]
        solid = moved & self._lane_solid[prev, lane]
        self._prev_lane, self._lane_valid = lane, torch.ones_like(self._lane_valid)
        return legal.float(), solid.float()

    def _add_crosswalk_priority(self, order, priority, yields):
        """Give a crossing right of way over the road lanelets that pass through it.

        The map relates road lanelets to each other and says nothing about pedestrians,
        so `failtoyield` could never fire for one - a car rolling through a crossing
        with somebody on it broke no rule the reward could see. A road lanelet whose
        centreline runs inside a crosswalk polygon is one that has to give way to it.
        """
        cross = [ll for ll in self.lanelet_map.laneletLayer
                 if 'subtype' in ll.attributes and ll.attributes['subtype'] == 'crosswalk'
                 and ll.id in order]
        n = 0
        for ll in self.lanelet_map.laneletLayer:
            if ll.id not in order or 'subtype' not in ll.attributes:
                continue
            if ll.attributes['subtype'] != 'road':
                continue
            pts = [(q.x, q.y) for q in ll.centerline]
            if len(pts) < 2:
                continue
            a = np.asarray(pts, dtype=np.float64)
            # resample so a short crossing is not stepped over
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
            if cum[-1] < 1e-6:
                continue
            want = np.arange(0.0, cum[-1], 1.0)
            xs = np.interp(want, cum, a[:, 0])
            ys = np.interp(want, cum, a[:, 1])
            hit = set()
            for x, y in zip(xs, ys):
                p = lanelet2.core.BasicPoint2d(float(x), float(y))
                for dist, cw in lanelet2.geometry.findNearest(
                        self.lanelet_map.laneletLayer, p, 8):
                    if dist <= 0.01 and cw.id in order and cw.id != ll.id:
                        if 'subtype' in cw.attributes and cw.attributes['subtype'] == 'crosswalk':
                            hit.add(cw.id)
            for cid in hit:
                priority[order[ll.id], order[cid]] = True
                yields[order[ll.id]] = True
                n += 1
        return n

    def _add_oncoming_turn_priority(self):
        """Make right-turning lanelets give way to the oncoming traffic they cross.

        The map does not describe this conflict: right_of_way marks 209 give-way
        lanelets but never pairs a right turn with the oncoming lane it cuts across, and
        the synthetic signal phases cannot separate them either - opposing approaches
        are parallel, so they share a phase and are green together (228 of 460
        same-phase stop-line pairs in a junction are head-on). Japan drives on the left,
        so the right turn is the one that crosses; left turns conflict with the
        crosswalk instead, which is deliberately left unregulated here.

        The pairing is geometric - a right-turn centreline passing within
        TURN_CONFLICT_DIST of an oncoming centreline - rather than taken from
        turn_direction alone, so it only fires where the paths actually cross.
        """
        order = self._lanelet_order
        lls = [ll for ll in self.lanelet_map.laneletLayer if ll.id in order]
        pts, head, cent = {}, {}, {}
        for ll in lls:
            a = np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64)
            if a.shape[0] < 2:
                continue
            pts[ll.id] = a
            d = a[-1] - a[0]
            head[ll.id] = np.arctan2(d[1], d[0])
            cent[ll.id] = a.mean(axis=0)
        right = [ll.id for ll in lls if 'turn_direction' in ll.attributes
                 and ll.attributes['turn_direction'] == 'right' and ll.id in pts]
        added = 0
        for r in right:
            for o in pts:
                if o == r:
                    continue
                if np.hypot(*(cent[o] - cent[r])) > self.TURN_CONFLICT_RADIUS:
                    continue
                if np.cos(head[o] - head[r]) > -0.3:          # not oncoming
                    continue
                gap = np.linalg.norm(pts[r][:, None, :] - pts[o][None, :, :], axis=-1).min()
                if gap < self.TURN_CONFLICT_DIST:
                    self._priority[order[r], order[o]] = True
                    self._yield_lanelet[order[r]] = True
                    added += 1
        if right and not added:
            raise RuntimeError('right-turn lanelets found but no oncoming conflicts paired')
        self._turn_priority_pairs = added

    def _agent_lanelet(self, state):
        """Index (into the right-of-way tables) of the lanelet each agent is on."""
        return self._pt_lanelet[self._nearest_rule_point(state)]

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
        seen, sig, owners = {}, [], []
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
                owners.append(ll)
        n_vehicle = len(sig)
        sig, owners = self._add_crosswalk_signals(sig, owners)
        if not sig:
            self._sl_xy = torch.zeros(0, 2, device=self.device)
            self._sl_dir = torch.zeros(0, device=self.device)
            self._sl_group = torch.zeros(0, dtype=torch.long, device=self.device)
            self._sl_offset = torch.zeros(0, dtype=torch.long, device=self.device)
            self._sl_ped = torch.zeros(0, dtype=torch.bool, device=self.device)
            self._sl_allowed = torch.zeros(self.NUM_TYPES, 0, dtype=torch.bool, device=self.device)
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

        # A crossing takes the opposite phase to the traffic it actually cuts across,
        # read off those approaches rather than inferred from its own heading: at a
        # skewed junction the parallel/perpendicular split lands on the wrong side of
        # its 45-degree boundary, and 26 of 139 crossings came out in phase with the
        # cars they cross - a green light walking into moving traffic.
        n_vehicle = min(n_vehicle, len(sig))
        veh = np.arange(n_vehicle)
        for i in range(n_vehicle, len(sig)):
            # by distance, not by cluster: a crossing sits at the mouth of the junction
            # and the greedy clustering often puts it in a cluster of its own
            near = veh[np.linalg.norm(sig[veh, :2] - sig[i, :2], axis=1)
                       < self.INTERSECTION_RADIUS]
            crossed = near[np.abs(np.cos(sig[near, 2] - sig[i, 2])) < 0.5]
            if crossed.size:
                # pair with the nearest crossed approach and take both its cycle and the
                # opposite of its phase. Pairing against all of them does not always have
                # an answer: at 20 of 139 crossings the traffic being cut across is itself
                # split between the two groups, so no single pedestrian phase is antiphase
                # to all of it. That is the two-phase model running out, not a bug.
                nearest = crossed[np.argmin(
                    np.linalg.norm(sig[crossed, :2] - sig[i, :2], axis=1))]
                group[i] = 1 - group[nearest]
                offset[i] = offset[nearest]
        self._sl_xy = torch.tensor(sig[:, :2], dtype=torch.float32, device=self.device)
        self._sl_dir = torch.tensor(sig[:, 2], dtype=torch.float32, device=self.device)
        self._sl_group = torch.tensor(group, device=self.device)
        self._sl_offset = torch.tensor(offset, device=self.device)
        ped = np.zeros(len(sig), dtype=bool)
        ped[n_vehicle:] = True
        self._sl_ped = torch.tensor(ped, device=self.device)
        self._sl_allowed = self._signal_participant_mask(owners)

    def _add_crosswalk_signals(self, sig, owners):
        """Give the crosswalks at a signalised junction a phase of their own.

        Without this there is no pedestrian signal anywhere: every stop line in the map
        belongs to a `road` lanelet, so pedestrians crossed whenever they liked and a car
        met one on green as often as on red (measured: 50.6% green at the moment a car
        first touched a pedestrian, against a 42.9% base rate).

        A crossing is entered from either end, so each crosswalk contributes two stop
        lines pointing inwards. The phase comes out of the *same* rule the vehicle
        approaches use - group by direction relative to the junction's reference - which
        is right by construction: a pedestrian crossing a road walks parallel to the
        cross street, so grouping on their own heading puts them on the cross street's
        green, exactly when the traffic they would meet is stopped.

        Only crosswalks near an existing vehicle stop line get one. An unsignalised
        crossing mid-block has no light in the map and must not gain a synthetic one.
        """
        if not sig:
            return sig, owners
        veh_xy = np.asarray([[s[0], s[1]] for s in sig], dtype=np.float64)
        for ll in self.lanelet_map.laneletLayer:
            if 'subtype' not in ll.attributes or ll.attributes['subtype'] != 'crosswalk':
                continue
            cl = np.asarray([(q.x, q.y) for q in ll.centerline], dtype=np.float64)
            if cl.shape[0] < 2:
                continue
            if np.linalg.norm(veh_xy - cl.mean(axis=0), axis=1).min() > self.INTERSECTION_RADIUS:
                continue                                   # unsignalised crossing
            for entry, nxt in ((cl[0], cl[1]), (cl[-1], cl[-2])):
                d = nxt - entry
                sig.append((entry[0], entry[1], float(np.arctan2(d[1], d[0]))))
                owners.append(ll)
        return sig, owners

    def _signal_participant_mask(self, owners):
        """[NUM_TYPES, S] mask of which stop lines each type is judged against. The base
        env has vehicles only, so every line applies to it."""
        return torch.ones(self.NUM_TYPES, len(owners), dtype=torch.bool, device=self.device)

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
        # only a stop line in front, within range, facing the way the agent drives, and
        # meant for this participant - a pedestrian judged against a car's stop line was
        # 49% of all red-light violations, and none of them meant anything
        aligned = torch.cos(psi.unsqueeze(1) - self._sl_dir.unsqueeze(0)) > 0.5
        valid = (aligned & (ahead > 0) & (dist < self.TL_RADIUS)
                 & self._sl_allowed[self._agent_types()])
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

    def _proximity(self, state):
        """Risk of the interaction with the nearest conflicting agent.

        One measure cannot cover every conflict. The traffic-safety literature splits
        them - TTC suits rear-end, PET suits crossing - and the RSS-derived RL objective
        in arXiv:2505.06737 splits the reward the same way, into same-direction,
        opposite-direction and intersecting. Our own collisions split 50.8 / 21.3 / 27.9
        across exactly those three, so each gets the term that fits:

        - same direction: RSS longitudinal safe distance, which asks whether the ego
          could still stop if the car ahead braked hard. Plain TTC cannot see this at
          all - matched speeds give infinite TTC at any gap, so tailgating was free.
        - opposite direction: the same, with both vehicles' stopping distances summed.
        - crossing: time to predicted contact from |p + w t| = R.

        Everything is measured in the ego frame, so lateral offset suppresses conflicts
        that are really passes. That matters: judged on range and range-rate alone, a
        parked car one lane over scored 0.490 against 0.497 for one in our own lane, and
        an oncoming car in the next lane 0.554 - the worst of all, since closing speed
        adds. On a two-way street that was a standing charge for meeting any traffic.
        """
        A = state.shape[0]
        xy, psi, v = state[:, :2], state[:, 2], state[:, 3].abs()
        c, sn = torch.cos(psi), torch.sin(psi)
        rel = xy.unsqueeze(0) - xy.unsqueeze(1)                    # [A, A, 2] j - i
        dx = rel[..., 0] * c.unsqueeze(1) + rel[..., 1] * sn.unsqueeze(1)   # ahead of i
        dy = -rel[..., 0] * sn.unsqueeze(1) + rel[..., 1] * c.unsqueeze(1)  # left of i
        size = self.simulator.get_agent_size()[0][..., :2]
        half_l, half_w = 0.5 * size[:, 0], 0.5 * size[:, 1]
        c_x = half_l.unsqueeze(0) + half_l.unsqueeze(1)
        c_y = half_w.unsqueeze(0) + half_w.unsqueeze(1) + self.PROXIMITY_MARGIN
        gap = (dx - c_x).clamp(min=0.0)                            # clear longitudinal gap
        ahead = dx > 0
        # Sharing a lane is decided by lanelet identity, not by lateral offset in the
        # ego's straight-line frame. On a bend the car ahead is offset sideways in that
        # frame - 3.7 m at 15 m ahead on a 30 m radius - and a geometric test loses it
        # entirely past the width threshold. Worse, the threshold is crossed sooner the
        # longer the required gap, so the term went blind exactly where it matters most:
        # approaching a stopped car at speed. Lanelet identity has no curvature term.
        # Union of two tests, because neither covers the other. The geometric one is
        # exact on a straight road but loses the car ahead round a bend - 3.7 m of
        # apparent lateral offset at 15 m ahead on a 30 m radius, and the threshold is
        # crossed sooner the longer the required gap, so it went blind precisely when
        # approaching a stopped car at speed. Lanelet identity has no curvature term but
        # is noisy where lanelets overlap: on this map two cars 6 m apart on one route
        # can be assigned to different, unconnected lanelets.
        lane = self._agent_lanelet(state)
        overlap = ((dy.abs() < c_y)
                   | (self._same_corridor(lane) & (dy.abs() < self.CORRIDOR_HALF_WIDTH)))

        rho, a_acc, a_brk = self.RSS_REACTION, self.RSS_ACCEL, self.RSS_BRAKE
        vi, vj = v.unsqueeze(1), v.unsqueeze(0)
        # distance the ego covers before it can stop, reacting for rho at a_acc first
        d_ego = vi * rho + 0.5 * a_acc * rho ** 2 + (vi + a_acc * rho) ** 2 / (2 * a_brk)
        same = torch.cos(psi.unsqueeze(0) - psi.unsqueeze(1)) > 0.7
        opposite = torch.cos(psi.unsqueeze(0) - psi.unsqueeze(1)) < -0.7
        r_same = (d_ego - vj ** 2 / (2 * a_brk)).clamp(min=c_x)    # lead brakes flat out
        r_opp = d_ego + (vj * rho + 0.5 * a_acc * rho ** 2
                         + (vj + a_acc * rho) ** 2 / (2 * a_brk))
        r_x = torch.where(same, r_same, r_opp)
        longitudinal = torch.relu(1 - gap / r_x.clamp(min=1e-3))
        longitudinal = longitudinal * (overlap & ahead & (same | opposite)).float()

        # crossing: first root of |p + w t| = R, no real root means the paths miss
        vel = torch.stack([c, sn], dim=-1) * v.unsqueeze(-1)
        w = vel.unsqueeze(0) - vel.unsqueeze(1)
        R = half_w.unsqueeze(0) + half_w.unsqueeze(1) + self.PROXIMITY_MARGIN
        if self.capsule_risk:
            # Measure from the nearest point of the ego's body, not its centre. Both
            # agents were discs of their half *width*, which turns a 4.97 m car into a
            # 1.02 m circle: a pedestrian stepping out two metres in front of the bumper
            # is 4.5 m from that circle, so the time-to-contact is computed to a point
            # the car reaches long after it has already hit them. The body is a capsule
            # of its own length instead - the same radius, slid along the centre line.
            s_ego = dx.clamp(-half_l.unsqueeze(1), half_l.unsqueeze(1))
            nose = torch.stack([s_ego * c.unsqueeze(1) - 0.0 * sn.unsqueeze(1),
                                s_ego * sn.unsqueeze(1)], dim=-1)
            rel = rel - nose
        qa = (w * w).sum(-1)
        qb = 2.0 * (rel * w).sum(-1)
        qc = (rel * rel).sum(-1) - R * R
        disc = qb * qb - 4 * qa * qc
        inf = torch.full_like(qa, float('inf'))
        t = torch.where((disc > 0) & (qa > 1e-6),
                        (-qb - disc.clamp(min=0).sqrt()) / (2 * qa.clamp(min=1e-6)), inf)
        ttc = torch.where(t > 0, t, inf)
        crossing = torch.relu(1 - ttc / self.TTC_THRESHOLD) * (~(same | opposite)).float()

        risk = torch.maximum(longitudinal, crossing)
        eye = torch.eye(A, dtype=torch.bool, device=self.device)
        mask = self.simulator.get_present_mask()[0]
        risk = torch.where(eye | ~mask.unsqueeze(0), torch.zeros_like(risk), risk)
        if not self.proximity_per_type:
            return risk.max(dim=1).values * mask
        kinds = self._agent_types()
        worst = torch.zeros(A, device=self.device)
        for t in range(self.NUM_TYPES):
            worst = worst + torch.where(kinds.unsqueeze(0) == t, risk,
                                        torch.zeros_like(risk)).max(dim=1).values
        return worst * mask

    def _same_corridor(self, lane):
        """[A, A] mask of agent pairs travelling the same stretch of road.

        True when both are on the same lanelet, or when one is directly downstream of
        the other, so a car round the next bend still counts as being in front.
        """
        same = lane.unsqueeze(0) == lane.unsqueeze(1)
        return same | self._downstream[lane.unsqueeze(1), lane.unsqueeze(0)]

    def _lane_offset(self, state):
        """How far each agent has strayed from the centre of a lane it may use, as 0 at
        the tolerance and 1 at LANE_SCALE beyond it. See W_LANE.

        Measured perpendicular to the lane, not as the distance to the nearest table
        point. The table runs at 1.38 m spacing, so the nearest point sits up to 0.7 m
        along the lane as well as across it, and that longitudinal error adds in
        quadrature: a median 0.95 m against a true 0.45 m of offset. Charging it made
        two thirds of the penalty an artefact of the sampling - agents paid about
        0.077 per step against a forward incentive of roughly 0.10, so driving slower to
        wander less was the rational response, and speed/limit fell from 0.65 to 0.53.

        Masked per participant, so a pedestrian is measured against the crosswalk it is
        on, not the road underneath. The agent can see what it is charged for: the road
        block gives it the same centreline points in its own frame.
        """
        near = self._nearest_rule_point(state)
        rel = state[:, :2] - self._rule_xy[near]
        d = self._rule_dir[near]
        lateral = (-torch.sin(d) * rel[:, 0] + torch.cos(d) * rel[:, 1]).abs()
        return ((lateral - self.LANE_TOLERANCE) / self.LANE_SCALE).clamp(0.0, 1.0)

    def _speed_limit(self, near):
        """The limit each agent is judged against at lane point `near`. One road-user
        type here, so it is simply the lane's own limit."""
        return self._rule_speed[near]

    def _speed_ratio(self, state):
        """Per-agent speed as a fraction of the limit where it is. Reported because
        every rule metric is a violation rate, and a policy that simply stops scores
        perfectly on all of them - which is what happened once the penalties were
        applied from the first update."""
        near = self._nearest_rule_point(state)
        return (state[:, 3].abs() / self._speed_limit(near)).clamp(max=3)

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
        idx, signed, _ = self._signals_cached(state)

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
        limit = self._speed_limit(near)
        excess = (state[:, 3].abs() - limit).clamp(min=0)             # m/s over the limit
        over = (excess - self.SPEED_TOLERANCE).clamp(min=0)
        speeding = over.pow(2).clamp(max=self.SPEED_PENALTY_CAP / max(self.w_speeding, 1e-9))

        # failure to yield: rolling through a give-way lanelet while an agent with
        # priority is close by. A proxy, not a conflict-point calculation - it says
        # "you should have waited", not "you would have hit them".
        lane = self._pt_lanelet[near]
        close = torch.cdist(state[:, :2], state[:, :2]) < self.YIELD_RADIUS
        close.fill_diagonal_(False)
        prio = self._priority[lane][:, lane]                     # [A, A] j has priority over i
        threatened = (close & prio).any(dim=1)
        failtoyield = (self._yield_lanelet[lane] & threatened
                       & (state[:, 3].abs() > self.YIELD_SPEED)).float()
        self._excess_kmh = excess * 3.6
        return redlight, wrongway, speeding, failtoyield

    def _nearest_rule_point(self, state):
        """Nearest lane point for each agent, restricted to the ones its participant
        type may legally be on.

        The observation and the rule check both need this, and it is a cdist against
        ~28k points, so the result is cached for the state tensor it was computed from
        (three calls per step is what took the step from 30 to 60 ms).
        """
        key = state.data_ptr(), state._version
        if getattr(self, '_near_key', None) != key:
            d = torch.cdist(state[:, :2], self._rule_xy)
            d = d.masked_fill(~self._rule_allowed[self._agent_types()], float('inf'))
            self._near_key, self._near_val = key, d.argmin(dim=1)
        return self._near_val

    def _signals_cached(self, state):
        key = state.data_ptr(), state._version
        if getattr(self, '_sig_key', None) != key:
            self._sig_key, self._sig_val = key, self._signals(state)
        return self._sig_val

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
    def _at_fault(self, state):
        """Contacts this agent started by driving into someone, as [A].

        The contact rate counts both parties of every touch, so an agent that is
        stationary when someone reaches it scores the same as the one that arrived -
        and 36% of contacts here have a stationary party. NAVSIM and
        arXiv:2606.19370 report *at-fault* collisions instead, contact caused by the
        agent being scored. Fault is assigned to whichever party was closing on the
        other faster along the line of centres at the moment contact began; if both
        were closing at the same rate, both are at fault.
        """
        pairs = self._contact_pairs
        new = pairs & ~self._prev_pairs
        self._prev_pairs = pairs
        if not bool(new.any()):
            return torch.zeros(state.shape[0], device=self.device)
        xy, psi, v = state[:, :2], state[:, 2], state[:, 3]
        rel = xy.unsqueeze(0) - xy.unsqueeze(1)                    # [A, A, 2] j - i
        n = rel / (rel.norm(dim=-1, keepdim=True) + 1e-6)          # unit i -> j
        vel = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1) * v.unsqueeze(-1)
        toward_j = (vel.unsqueeze(1) * n).sum(-1)                  # i closing on j
        toward_i = (vel.unsqueeze(0) * -n).sum(-1)                 # j closing on i
        return (new & (toward_j >= toward_i)).float().sum(dim=1)

    def _collision_weight(self):
        """[A] multiplier for hitting each agent. One road-user type here, so uniform."""
        return torch.ones(self.num_agents, device=self.device)

    def _collision(self, state):
        """Per-agent collision loss, `simulator.compute_collision()[0]` weighted by what
        was hit (see `_collision_weight`)."""
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
        # what you hit matters: running into a pedestrian and clipping a kerb cost the
        # same before, and 156 of the 782 contacts in an episode were a car reaching a
        # pedestrian. Columns are weighted by the *other* agent's vulnerability, so the
        # heavier party pays more for the same event.
        overlap = overlap * self._collision_weight().unsqueeze(0)
        self._contact_pairs = overlap > 0        # [A, A]; consumed by _at_fault
        return overlap.sum(dim=-1) * mask

    def _steering_params(self):
        """Per-agent rear-axle distance, steering limit and 'is a bicycle model' mask,
        used by `_limit_steering`. One type here, so all three are uniform."""
        A = self.num_agents
        return (torch.full((A,), self.lr, device=self.device),
                torch.full((A,), self.MAX_STEER * self.max_steer_scale, device=self.device),
                torch.ones(A, dtype=torch.bool, device=self.device))

    def _limit_steering(self, action, state):
        """Hold the steering command to what the vehicle could actually do.

        Acts on the command rather than on the state afterwards, so the position
        integration never sees a slip angle the tyres could not hold. See MAX_STEER /
        LAT_ACCEL_MAX / STEER_RATE_MAX for why each of the three limits is here.
        """
        lr, steer_max, wheeled = self._steer_lr, self._steer_max, self._steer_wheeled
        beta = action[:, 1] * steer_max                             # command, radians
        # a_lat = v^2 / lr * sin(beta); solve for the largest beta that stays under it
        v2 = state[:, 3].pow(2).clamp(min=1e-6)
        sin_cap = (self.LAT_ACCEL_MAX * lr / v2).clamp(max=1.0)
        cap = torch.minimum(torch.asin(sin_cap), steer_max)
        beta = torch.max(torch.min(beta, cap), -cap)
        # and it may only move so fast from where it was
        d = self.STEER_RATE_MAX * self.dt
        beta = torch.max(torch.min(beta, self._prev_steer + d), self._prev_steer - d)
        self._prev_steer = torch.where(wheeled, beta, self._prev_steer)
        out = action.clone()
        out[:, 1] = torch.where(wheeled, beta / steer_max, action[:, 1])
        return out

    def _build_simulator(self):
        A = self.num_agents
        kin = SteeringLimitedBicycle(dt=self.dt)
        kin.set_params(lr=torch.full((1, A), self.lr, device=self.device))
        kin.set_steering_limit(self._steering_params()[1])
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
        # the ego block is normalised by each agent's own goal spacing, so the features
        # stay in range whatever that spacing is - see _ego_features
        self._goal_scale = torch.as_tensor(self._goal_dist, dtype=torch.float32,
                                           device=self.device).clamp(min=1.0)
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

    def _recover(self, mask):
        """Put agents that have driven off the road and stayed off back on a route."""
        self._respawn_xy = self._state()[:, :2].tolist()   # one host read for the batch
        moved = []
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            pose = self._respawn(i)
            if pose is not None:
                moved.append((i, pose))
        self._lost_count = getattr(self, '_lost_count', 0) + len(moved)
        return self._teleport(moved)

    def _teleport(self, moved):
        """Move the listed agents to (x, y, heading) at rest. Returns whether any moved."""
        if not moved:
            return False
        state = self._state().clone()
        m = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        for i, (x, y, h) in moved:
            state[i, 0], state[i, 1], state[i, 2], state[i, 3] = x, y, h, 0.0
            m[i] = True
            self._prev_sl_idx[i] = -1
        # a teleport changes the overlap discontinuously; that jump is not the agent
        # driving into anything, so the next step must not charge it
        self._just_moved = self._just_moved | m
        self._teleported = self._teleported | m
        self.simulator.set_state(state.unsqueeze(0), mask=m.unsqueeze(0))
        return True

    def _advance_goals(self, mask):
        """Place the next goal further along the route for every agent in `mask`.

        The map is a finite cut-out, so a route eventually reaches its edge; rather
        than parking the agent there, it is respawned on a fresh route (which is why
        this returns whether any agent moved - the caller must re-read the state).
        """
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        self._respawn_xy = self._state()[:, :2].tolist()   # one host read for the batch
        moved = []
        for i in idx:
            s = self._s_goal[i] + self._goal_dist[i]
            total = float(self._route_cache[i][2][-1])
            if s > total - 1e-3:
                if self._extend_route(i):
                    total = float(self._route_cache[i][2][-1])
                else:
                    pose = None if self.despawn_at_goal else self._respawn(i)
                    if pose is None:   # nowhere to go: this was the last goal
                        self._route_exhausted[i] = True
                        self._any_finished = True
                        self._s_goal[i] = total
                        if self.despawn_at_goal:
                            self._despawn.append(i)
                        continue
                    moved.append((i, pose))
                    total = float(self._route_cache[i][2][-1])
                    s = min(self._goal_dist[i], total)
            self._s_goal[i] = min(s, total)
        self._sync_goals(idx)
        return self._teleport(moved)

    def _observation(self, state, prev_action):
        # [A, EGO_DIM | MAX_PARTNERS*PARTNER_FEATURES | MAX_ROAD*ROAD_FEATURES];
        # the policy slices off the ego part and max-pools the partner and road sets.
        return torch.cat([self._ego_features(state, prev_action),
                          self._partner_block(state),
                          self._road_block(state)], dim=-1)

    def _ego_features(self, state, prev_action):
        x, y, psi, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        near = self._nearest_rule_point(state)
        limit, lane = self._speed_limit(near), self._pt_lanelet[near]
        dx, dy = self.goals[:, 0] - x, self.goals[:, 1] - y
        c, s = torch.cos(psi), torch.sin(psi)
        gx_e = c * dx + s * dy            # goal in ego frame
        gy_e = -s * dx + c * dy
        dist = torch.linalg.norm(torch.stack([dx, dy], -1), dim=-1)
        head_err = torch.atan2(gy_e, gx_e)
        # Normalised by the agent's own goal spacing rather than a fixed 100 m / 50 m.
        # A vehicle's goal sits 150 m away, so the old constants pinned all three of
        # these features at their clamp for 62% of its steps: distance, and both
        # components of the goal's position, were literally constant while it drove.
        gs = self._goal_scale
        base = torch.stack([
            v / 10.0,
            (dist / gs).clamp(max=2.0),
            torch.cos(head_err), torch.sin(head_err),
            (gx_e / gs).clamp(-2, 2), (gy_e / gs).clamp(-2, 2),
            prev_action[:, 0], prev_action[:, 1],
            limit / 10.0,
            self._yield_lanelet[lane].float(),      # "I have to give way here"
        ], dim=-1)
        # the signal block is what makes the red-light penalty learnable
        out = torch.cat([base, self._signals_cached(state)[2]], dim=-1)
        return torch.cat([out, self._branch_block(state)], dim=-1) if self.branch_obs else out

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
            # map-derived priority of each neighbour over the ego's lanelet. This is
            # position plus HD map, not intention - see W_YIELD.
            lane = self._agent_lanelet(state)
            has_prio = self._priority[lane][torch.arange(A, device=self.device)[:, None],
                                            lane[nidx]].float()
            feat = torch.stack([rel_x, rel_y, wj, lj,
                                torch.cos(rel_h), torch.sin(rel_h), vj, has_prio], dim=-1)
            extra = self._partner_extra(nidx, valid)
            if extra is not None:
                feat = torch.cat([feat, extra], dim=-1)
            feat = feat * valid[..., None]                      # zero out far/empty slots
            out[:, :k, :] = feat
        return out.reshape(A, -1)

    def _road_block(self, state):
        """Road-graph observation: one lane-centreline point per forward distance band,
        in the ego frame, zero-padded. Returns [A, MAX_ROAD * ROAD_FEATURES].

        GPUDrive takes the K nearest points, which was fine while the point cloud was
        coarse but gives no preview once it is dense - the K nearest are all within a
        few metres. Measured on the trained policy, the road block reached a median of
        15 m ahead, 1.5 s at 10 m/s, while slowing for a 9.4 m junction radius under the
        lateral-acceleration limit needs 2-3 s of warning. Banding the points by
        distance ahead guarantees the preview instead of hoping the sampling provides
        it: the same ten slots now describe the road from 5 m behind to ROAD_RADIUS
        ahead, one point per band, nearest within the band.
        """
        A = state.shape[0]
        K = self.MAX_ROAD
        out = torch.zeros(A, K, self.ROAD_FEATURES, device=self.device)
        M = self._road_xy.shape[0]
        if M == 0:
            return out.reshape(A, -1)
        psi = state[:, 2]
        dx = self._road_xy[:, 0][None, :] - state[:, 0][:, None]       # [A, M]
        dy = self._road_xy[:, 1][None, :] - state[:, 1][:, None]
        c, s = torch.cos(psi)[:, None], torch.sin(psi)[:, None]
        fwd = c * dx + s * dy                                          # ahead of the agent
        lat = -s * dx + c * dy
        dist2 = dx * dx + dy * dy
        # One scatter over all points rather than a masked min per band: looping the
        # bands costs K passes over [A, M] and, at the density banding needs, that was
        # 28 of the step's 43 ms. The distance and the point index are packed into one
        # integer so a single amin returns both.
        width = (self.ROAD_RADIUS + self.ROAD_BEHIND) / K
        bid = ((fwd + self.ROAD_BEHIND) / width).floor().long()
        keep = (bid >= 0) & (bid < K) & (dist2 < self.ROAD_RADIUS ** 2)
        code = (dist2 * 100).long() * M + torch.arange(M, device=self.device)[None, :]
        SENTINEL = torch.iinfo(torch.int64).max
        code = torch.where(keep, code, torch.full_like(code, SENTINEL))
        slot = torch.where(keep, bid, torch.full_like(bid, K))
        best = torch.full((A, K + 1), SENTINEL, dtype=torch.int64, device=self.device)
        best.scatter_reduce_(1, slot, code, reduce='amin', include_self=True)
        best = best[:, :K]                                             # [A, K]
        has = best != SENTINEL
        idx = (best % M).clamp(min=0)                                  # winning point per band
        rel_dir = self._road_dir[idx] - psi[:, None]
        out = torch.stack([torch.gather(fwd, 1, idx) / self.ROAD_RADIUS,
                           torch.gather(lat, 1, idx) / self.ROAD_RADIUS,
                           torch.cos(rel_dir), torch.sin(rel_dir)], dim=-1)
        return (out * has[..., None]).reshape(A, -1)

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
        self._steer_lr, self._steer_max, self._steer_wheeled = self._steering_params()
        self._build_simulator()
        self._t = 0
        self._prev_steer = torch.zeros(self.num_agents, device=self.device)
        self._prev_action = torch.zeros(self.num_agents, self.ACT_DIM, device=self.device)
        self._reached = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._goals_reached = torch.zeros(self.num_agents, device=self.device)
        self._route_exhausted = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._off_run = torch.zeros(self.num_agents, device=self.device)
        self._prev_collision = torch.zeros(self.num_agents, device=self.device)
        self._prev_pairs = torch.zeros(self.num_agents, self.num_agents,
                                       dtype=torch.bool, device=self.device)
        self._prev_lane = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        self._lane_valid = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._just_moved = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        if not hasattr(self, '_parked'):
            self._parked = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self._lost_count = 0
        self._any_finished = False
        self._despawn = []
        if self.despawn_at_goal:      # _build_simulator makes a fresh all-present mask
            self._n_despawned = 0
        self._prev_sl_idx = torch.full((self.num_agents,), -1, dtype=torch.long, device=self.device)
        self._prev_sl_signed = torch.zeros(self.num_agents, device=self.device)
        state = self._state()
        self._prev_dist = self._dist_to_goal(state)
        return self._observation(state, self._prev_action)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(-1, 1)
        was_reached = self._reached.clone()          # agents that already finished before this step
        self._teleported = torch.zeros(self.num_agents, dtype=torch.bool,
                                       device=self.device)
        action = self._limit_steering(action, self._state())
        # parked cars never act: they are scenery the policy has to drive around, and they
        # are excluded from training so their (meaningless) transitions carry no gradient
        action[was_reached | self._parked] = 0.0
        self.simulator.step(action.unsqueeze(0))
        self._post_physics()
        self._t += 1
        state = self._state()
        dist = self._dist_to_goal(state)

        progress = (self._prev_dist - dist)                                   # dense shaping
        collision = (self._collision(state) > 0).float()
        offroad = (self._offroad(state) > 0).float()
        proximity = self._proximity(state)
        if self.offroad_fix:
            off_cost = offroad.clamp(max=self.OFFROAD_CAP)
            progress = progress * (offroad <= 0).float()   # withheld, never reversed
        else:
            off_cost = offroad
            progress = progress * (1.0 - offroad)  # no credit for shortcutting off-road
        redlight, wrongway, speeding, failtoyield = self._rule_violations(state)
        lane = self._lane_offset(state)
        at_fault = self._at_fault(state)
        chg_legal, chg_solid = self._lane_change(state)
        # Charge driving *into* a contact, not sitting in one. The level counts a single
        # event for as many steps as the overlap lasts, so it is dominated by whoever is
        # stuck - 36% of contacts have a stationary party, and pedestrians clumping held
        # 1301 pair-steps against 81 onsets. The rise is the smooth analogue of an onset:
        # it integrates to how deep the agent drove in, and backing out is free.
        hit = (collision - self._prev_collision).clamp(min=0.0) * (~self._just_moved)
        self._prev_collision, self._just_moved = collision, torch.zeros_like(self._just_moved)
        newly_reached = (dist < self.goal_radius) & (~was_reached)
        # Forward motion is paid for either by closing on a goal point, or - when
        # w_follow is on - by covering ground while tracking one of the branches. The
        # second form has no bearing to a distant point in it at all, which is what made
        # the first one tell a car in lane to head sideways.
        if self.w_follow > 0:
            info_follow = self._follow_score(state)
            follow_term = self.w_follow * state[:, 3].abs() * self.dt * info_follow
            if self.offroad_fix:
                # the same withholding `progress` gets. Without it the follow reward is
                # paid in full off the road, which is the one path the off-road term
                # never reached - and the corridor run's `lost` went 24 -> 66/86.
                follow_term = follow_term * (offroad <= 0).float()
            forward = torch.where(self._branch_types(), follow_term,
                                  self.w_progress * progress)
        else:
            info_follow = torch.zeros_like(progress)
            forward = self.w_progress * progress
        reward = (forward - self.w_collision * hit
                  - self.w_collision_level * collision
                  - self.w_lanechange * chg_legal - self.w_solidcross * chg_solid
                  - self.w_offroad * off_cost + self.w_goal * newly_reached.float()
                  - self.w_redlight * redlight - self.w_wrongway * wrongway
                  - self.w_speeding * speeding            # see W_SPEEDING
                  - self.w_yield * failtoyield - self.w_proximity * proximity
                  - self.w_lane * lane)
        reward = torch.where(was_reached, torch.zeros_like(reward), reward)   # finished agents get 0

        # a sustained excursion means the agent is lost, not clipping a kerb
        self._off_run = (self._off_run + offroad) * offroad
        lost = self._off_run >= self.OFFROAD_PATIENCE
        # one fused read for the three rare events below, instead of one each
        flags = torch.stack([lost.any(), newly_reached.any()]).tolist()
        if self.rolling_goals and flags[0]:
            if self._recover(lost):
                state = self._state()
                dist = self._dist_to_goal(state)
            self._off_run = self._off_run * (~lost).float()
        self._despawn = []
        self._goals_reached += newly_reached.float()
        if self.rolling_goals:
            # Hand out the next goal instead of parking the agent. Only an agent whose
            # route cannot be extended any further is finished.
            if flags[1]:
                if self._advance_goals(newly_reached):
                    state = self._state()      # some agents were respawned elsewhere
                # the goal moved, so re-measure: otherwise the jump in distance would be
                # charged to the next step as a large negative `progress`
                dist = self._dist_to_goal(state)
            self._reached = was_reached | (newly_reached & self._route_exhausted)
        else:
            self._reached = was_reached | (dist < self.goal_radius)
        if self._despawn:
            # out of the collision, proximity and off-road sums, and out of the picture
            m = self.simulator.get_present_mask().clone()
            for i in self._despawn:
                m[0, i] = False
            self.simulator.present_mask = m
        # Freeze finished agents in place so they stop moving and don't drift into others.
        # `_any_finished` is maintained on the host by _advance_goals, so the freeze
        # check needs no device read at all - with rolling goals it is almost never set
        if self._any_finished:
            frozen = state.clone(); frozen[self._reached, 3] = 0.0
            self.simulator.set_state(frozen.unsqueeze(0), mask=self._reached.unsqueeze(0))
            state = self._state()
        self._prev_dist = dist
        self._prev_action = action
        done = self._reached.clone() | (self._t >= self.max_steps)
        if self.terminate_on_teleport:
            done = done | self._teleported
        # `active` marks transitions that count for training: an agent's steps are
        # valid up to and including the step it reaches the goal, then excluded.
        active = ~was_reached & ~self._parked
        # Metrics stay as 0-dim tensors. Every float() here is a device-to-host copy that
        # flushes the CUDA queue, and with ~25 of them per step (12 of which are per-type)
        # the step spent 42 of its 71 ms waiting on synchronisation - 92 syncs per step,
        # against 0.8% of device time in actual transfers. The trainer stacks these and
        # converts once per rollout instead.
        am = active.float()
        n_active = am.sum().clamp(min=1)
        mean_active = lambda x: (x * am).sum() / n_active
        info = {
            'reached': self._reached.float().mean(),
            'goals': self._goals_reached.mean(),          # goals collected per agent
            'lost': self._lost_count,                     # python int, no sync
            'present': self.simulator.get_present_mask()[0].float().mean(),
            'proximity': mean_active(proximity),
            # mean speed against the limit. Without this in the log a policy that has
            # simply stopped reads as perfect on every rule metric - which is exactly
            # what happened once the penalties were raised from the start of training.
            'speed_ratio': self._speed_ratio(state).mean(),
            'redlight': redlight.sum(),                   # crossings on red this step
            'wrongway': mean_active(wrongway),
            # violation = beyond the tolerated margin; excess is reported in km/h
            'speeding': mean_active((self._excess_kmh > self.SPEED_TOLERANCE * 3.6).float()),
            'speed_excess': mean_active(self._excess_kmh),
            'failtoyield': mean_active(failtoyield),
            # the level stays in the log so runs before the change remain comparable;
            # `contact` is what the reward now charges
            'collision': mean_active(collision),
            'contact': mean_active(hit),
            # `offroad` is metres of overhang, which reads nothing like a rate; both are
            # reported now because the two have been confused for each other
            'offroad': mean_active(offroad),
            'offroad_rate': mean_active((offroad > 0).float()),
            'lane': mean_active(lane),
            'follow': mean_active(info_follow),
            'atfault': at_fault.sum(),        # contacts this agent drove into, per step
            'lanechange': mean_active(chg_legal),
            'solidcross': mean_active(chg_solid),
            'active': active,
            # fixed-length episodes: the boundary is known without reading the device
            'episode_end': self._t >= self.max_steps,
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
        # the renderer does not consult present_mask on its own
        rmask = self.simulator.get_present_mask().unsqueeze(1)
        img = self.simulator.render(camera_xy=cam, camera_psi=psi,
                                    res=Resolution(self.render_res, self.render_res),
                                    fov=fov, rendering_mask=rmask)
        return img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
