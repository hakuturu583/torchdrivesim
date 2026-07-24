"""
Traffic simulation on an AWSIM / Autoware Lanelet2 map in TorchDriveSim.

The maps used by the AWSIM Quick Start demo
(https://autowarefoundation.github.io/AWSIM/GettingStarted/QuickStartDemo/) and
by Autoware in general are exported by Vector Map Builder (``generator="VMB"``).
That dialect differs from the CARLA maps bundled with TorchDriveSim in two ways
that break the stock loader, both handled here:

  1. Coordinates are geo-referenced (real lat/lon around a Japanese origin).
     TorchDriveSim's default ``UtmProjector(Origin(0, 0))`` puts the map in UTM
     zone 31 (centred on 3 deg E), which rejects longitudes near 140 deg E.
     Fix: project from a point inside the map (see ``map_latlon_origin``).

  2. The maps carry Autoware-specific regulatory elements (``detection_area``,
     ``no_stopping_area``, ``crosswalk``, ``virtual_traffic_light``, ...) that
     upstream lanelet2 does not implement, so a strict ``load`` throws.
     Fix: ``load_lanelet_map(..., robust=True)`` skips them and keeps geometry.

Autoware uses a right-handed ENU frame, so - unlike the CARLA maps - the map is
NOT inverted and lane markings are built with ``left_handed=False``.

Get a map first, e.g. the Autoware sample map (same VMB dialect as AWSIM):
    curl -L -o sample_map.osm \\
      https://raw.githubusercontent.com/tier4/autoware_lanelet2_map_validator/main/autoware_lanelet2_map_validator/test/data/map/sample_map.osm

Then run:
    python examples/awsim_lanelet2_traffic.py map_path=sample_map.osm
"""
import os
import re
import sys
import math
from dataclasses import dataclass

import numpy as np
import imageio
import torch
from omegaconf import OmegaConf

import lanelet2
from lanelet2.traffic_rules import Locations, Participants

from torchdrivesim.lanelet2 import (
    load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh,
)
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.kinematic import TeleportingKinematicModel
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.utils import Resolution


@dataclass
class AWSIMTrafficConfig:
    map_path: str = "sample_map.osm"
    save_dir: str = "./awsim_output"
    device: str = "cpu"
    agent_count: int = 8
    steps: int = 60
    dt: float = 0.2
    res: int = 800
    fov: float = 170.0


def _attr(lanelet, key, default=""):
    return lanelet.attributes[key] if key in lanelet.attributes else default


def map_latlon_origin(osm_path):
    """Mean lat/lon of all nodes; used to select the correct UTM zone."""
    txt = open(osm_path).read()
    lat = [float(v) for v in re.findall(r'lat="([-\d.]+)"', txt)]
    lon = [float(v) for v in re.findall(r'lon="([-\d.]+)"', txt)]
    if not lat:
        return (0.0, 0.0)
    return sum(lat) / len(lat), sum(lon) / len(lon)


def build_driving_surface_mesh(lanelet_map):
    road_mesh = road_mesh_from_lanelet_map(lanelet_map)
    road_mesh = BirdviewMesh.set_properties(road_mesh, category='road').to(road_mesh.device)
    lane_mesh = lanelet_map_to_lane_mesh(lanelet_map, left_handed=False)
    return lane_mesh.merge(road_mesh)


def build_route(graph, start, max_lanelets=14, max_len=260.0):
    """Chain following lanelets into a route and return its densified centerline."""
    chain, seen, total, cur = [start], {start.id}, 0.0, start
    while len(chain) < max_lanelets and total < max_len:
        total += lanelet2.geometry.length(cur.centerline)
        nxts = [l for l in graph.following(cur) if l.id not in seen]
        if not nxts:
            break
        cur = nxts[0]
        chain.append(cur)
        seen.add(cur.id)
    pts = []
    for ll in chain:
        for p in ll.centerline:
            xy = (p.x, p.y)
            if not pts or abs(pts[-1][0] - xy[0]) + abs(pts[-1][1] - xy[1]) > 1e-6:
                pts.append(xy)
    return np.asarray(pts, dtype=np.float64)


def resample_route(polyline, speed, dt, steps):
    """(steps+1, 3) array of (x, y, heading) advancing at constant speed along a polyline."""
    seg = np.diff(polyline, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    out = []
    for k in range(steps + 1):
        d = min(k * speed * dt, total - 1e-3)
        i = max(0, min(int(np.searchsorted(cum, d) - 1), len(seg) - 1))
        r = (d - cum[i]) / max(seglen[i], 1e-6)
        x, y = polyline[i] + r * seg[i]
        out.append((x, y, math.atan2(seg[i, 1], seg[i, 0])))
    return np.asarray(out)


def run(cfg: AWSIMTrafficConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device
    torch.manual_seed(0)

    # 1) Load the AWSIM/Autoware map the way Autoware itself does: the geo origin
    #    only selects the UTM zone, while the authoritative planar coordinates
    #    come from the local_x/local_y node tags (equivalent to Autoware's MGRS
    #    projector). recenter=True brings the large MGRS offsets back to origin.
    origin = map_latlon_origin(cfg.map_path)
    lanelet_map = load_lanelet_map(cfg.map_path, origin=origin, robust=True,
                                   use_local_coordinates=True, recenter=True)
    print(f"[map] {cfg.map_path} origin={origin} lanelets={len(list(lanelet_map.laneletLayer))}")

    # 2) Driving-surface mesh.
    mesh = build_driving_surface_mesh(lanelet_map).to(device)

    # 3) Lane-following routes via the lanelet2 routing graph.
    rules = lanelet2.traffic_rules.create(Locations.Germany, Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(lanelet_map, rules)
    roads = sorted([l for l in lanelet_map.laneletLayer if _attr(l, 'subtype') == 'road'],
                   key=lambda l: l.id)
    routes, i = [], 0
    while len(routes) < cfg.agent_count and i < len(roads):
        r = build_route(graph, roads[(i * 7) % len(roads)])
        if r.shape[0] >= 2 and np.hypot(*(r[-1] - r[0])) > 8.0:
            routes.append(r)
        i += 1
    if not routes:
        raise RuntimeError("No drivable routes found in the map.")
    print(f"[routes] {len(routes)} lane-following routes")

    speeds = [4.0 + 0.6 * j for j in range(len(routes))]
    trajs = np.stack([resample_route(r, s, cfg.dt, cfg.steps) for r, s in zip(routes, speeds)], axis=0)
    A = trajs.shape[0]

    init = torch.zeros(1, A, 4, device=device)
    init[0, :, :3] = torch.tensor(trajs[:, 0, :], dtype=torch.float32)
    init[0, :, 3] = torch.tensor(speeds, dtype=torch.float32)

    # 4) Drive agents through the simulator (teleporting kinematics = replay of routes).
    agent_size = torch.tensor([4.97, 2.04], device=device).view(1, 1, 2).expand(1, A, 2).contiguous()
    kinematic = TeleportingKinematicModel()
    kinematic.set_state(init)
    sim_cfg = TorchDriveConfig(left_handed_coordinates=False,
                               renderer=RendererConfig(left_handed_coordinates=False))
    renderer = renderer_from_config(sim_cfg.renderer)
    simulator = Simulator(
        cfg=sim_cfg, road_mesh=mesh, kinematic_model=kinematic, agent_size=agent_size,
        initial_present_mask=torch.ones(1, A, dtype=torch.bool, device=device),
        renderer=renderer, lanelet_map=[lanelet_map],
    )

    vx, vy = mesh.verts[..., 0], mesh.verts[..., 1]
    camera_xy = torch.tensor([[[float((vx.min() + vx.max()) / 2),
                                float((vy.min() + vy.max()) / 2)]]], device=device)
    camera_psi = torch.zeros(1, 1, 1, device=device)
    res = Resolution(cfg.res, cfg.res)

    frames, max_collision, max_offroad = [], 0.0, 0.0
    for t in range(cfg.steps + 1):
        image = simulator.render(camera_xy=camera_xy, camera_psi=camera_psi, res=res, fov=cfg.fov)
        frames.append(image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        max_collision = max(max_collision, float(simulator.compute_collision().max()))
        max_offroad = max(max_offroad, float(simulator.compute_offroad().max()))
        if t < cfg.steps:
            nxt = torch.zeros(1, A, 4, device=device)
            nxt[0, :, :3] = torch.tensor(trajs[:, t + 1, :], dtype=torch.float32)
            nxt[0, :, 3] = torch.tensor(speeds, dtype=torch.float32)
            simulator.step(nxt)

    gif_path = os.path.join(cfg.save_dir, "awsim_traffic.gif")
    imageio.mimsave(gif_path, frames, duration=cfg.dt, loop=0)
    imageio.imsave(os.path.join(cfg.save_dir, "awsim_map.png"), frames[0])
    print(f"[sim] {cfg.steps} steps, {A} agents -> {gif_path}")
    print(f"[metrics] max_collision={max_collision:.3f} max_offroad={max_offroad:.3f}")


if __name__ == '__main__':
    cli_cfg: AWSIMTrafficConfig = OmegaConf.structured(
        AWSIMTrafficConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    run(cli_cfg)  # type: ignore
