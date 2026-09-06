"""RA-UFE: Return-Aware Utility Frontier Exploration.

Pure-numpy grid algorithms with no rclpy dependency, so they can be unit
tested and profiled without a running ROS graph -- the same split the
provided simulator uses between sim_core.py and sim_node.py.

Pipeline: Map -> Frontiers -> Score frontiers -> Navigate -> Re-evaluate.

Cell classification follows the three-class convention:

    free      : 0 <= M(x, y) < T_f
    occupied  : M(x, y) > T_o
    unknown   : M(x, y) == -1

The band [T_f, T_o] is "known but not confidently free" -- it is neither
free nor occupied, so it is folded into UNKNOWN. That is the conservative
reading: we never drive through such a cell as if it were free, and it
still counts as something worth looking at.

Every candidate viewpoint is scored with

    U_i = w_I * I_i - w_P * C_i - w_H * H_i - w_R * R_i - w_D * D_i
where:

I_i: expected information gain;
C_i: path cost to frontier;
H_i: estimated future cost from frontier to home;
R_i: revisit/redundancy penalty;
D_i: direction-change penalty.
where each term is min-max normalised across the candidate set so the
weights are dimensionless and comparable.

"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Cell classes used internally (not the ROS occupancy values).
FREE = np.uint8(0)
OCC = np.uint8(1)
UNK = np.uint8(2)

_SQRT2 = math.sqrt(2.0)
# 8-connected neighbourhood as (dr, dc, geometric step in cell units).
_NEIGHBOURS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)


# --------------------------------------------------------------- map coding


def classify(data: np.ndarray, free_thresh: int = 25, occ_thresh: int = 65) -> np.ndarray:
    """nav_msgs occupancy values (-1, 0..100) -> FREE / OCC / UNK codes."""
    codes = np.full(data.shape, UNK, dtype=np.uint8)
    codes[(data >= 0) & (data < free_thresh)] = FREE
    codes[data > occ_thresh] = OCC
    return codes


def downsample(codes: np.ndarray, k: int) -> np.ndarray:
    """Block-reduce by k, conservatively: a coarse cell is OCC if any child
    is OCC, else UNK if any child is UNK, else FREE.

    Walls therefore grow by at most one coarse cell rather than dissolving,
    and a frontier survives downsampling instead of being averaged away.
    Padding is UNK, which is what "beyond the mapped area" actually means.
    """
    if k <= 1:
        return codes
    h, w = codes.shape
    ph, pw = (-h) % k, (-w) % k
    if ph or pw:
        codes = np.pad(codes, ((0, ph), (0, pw)), constant_values=UNK)
    blocks = codes.reshape(codes.shape[0] // k, k, codes.shape[1] // k, k)
    has_occ = (blocks == OCC).any(axis=(1, 3))
    has_unk = (blocks == UNK).any(axis=(1, 3))
    out = np.full(has_occ.shape, FREE, dtype=np.uint8)
    out[has_unk] = UNK
    out[has_occ] = OCC
    return out


def obstacle_distance(codes: np.ndarray) -> np.ndarray:
    """Chamfer distance (in cell units) from every cell to the nearest OCC
    cell. Two sequential passes with 1 / sqrt(2) weights -- close enough to
    Euclidean for clearance tests, and it avoids a scipy dependency (the
    container has numpy but no scipy).

    Unknown cells are *not* obstacles here: clearance measures distance to
    known walls only, so a viewpoint may sit next to unexplored space. That
    matches Nav2's `allow_unknown: true` planner setting.
    """
    h, w = codes.shape
    big = float(h + w) * 2.0
    d = np.where(codes == OCC, 0.0, big)

    for r in range(h):
        row = d[r]
        if r > 0:
            up = d[r - 1]
            cand = up + 1.0
            cand[:-1] = np.minimum(cand[:-1], up[1:] + _SQRT2)
            cand[1:] = np.minimum(cand[1:], up[:-1] + _SQRT2)
            np.minimum(row, cand, out=row)
        for c in range(1, w):
            v = row[c - 1] + 1.0
            if v < row[c]:
                row[c] = v

    for r in range(h - 1, -1, -1):
        row = d[r]
        if r < h - 1:
            dn = d[r + 1]
            cand = dn + 1.0
            cand[:-1] = np.minimum(cand[:-1], dn[1:] + _SQRT2)
            cand[1:] = np.minimum(cand[1:], dn[:-1] + _SQRT2)
            np.minimum(row, cand, out=row)
        for c in range(w - 2, -1, -1):
            v = row[c + 1] + 1.0
            if v < row[c]:
                row[c] = v

    return d


# ------------------------------------------------------------ frontier work


def frontier_mask(codes: np.ndarray) -> np.ndarray:
    """FREE cells with at least one 4-connected UNK neighbour."""
    free = codes == FREE
    unk = codes == UNK
    adj = np.zeros_like(unk)
    adj[:-1, :] |= unk[1:, :]
    adj[1:, :] |= unk[:-1, :]
    adj[:, :-1] |= unk[:, 1:]
    adj[:, 1:] |= unk[:, :-1]
    return free & adj


def cluster_cells(mask: np.ndarray, min_size: int = 3) -> list[np.ndarray]:
    """8-connected components of a boolean mask, as (N, 2) row/col arrays,
    largest first. Components smaller than min_size are dropped -- they are
    usually SLAM speckle on a wall, not real unexplored openings."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out: list[np.ndarray] = []
    rows, cols = np.nonzero(mask)
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if seen[r0, c0]:
            continue
        comp = []
        seen[r0, c0] = True
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc, _ in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        if len(comp) >= min_size:
            out.append(np.array(comp, dtype=np.int64))
    out.sort(key=len, reverse=True)
    return out


# ------------------------------------------------------------------ search


def dijkstra(
    traversable: np.ndarray,
    step_cost: np.ndarray,
    sources: list[tuple[int, int]],
    resolution: float,
) -> np.ndarray:
    """8-connected Dijkstra from `sources` over `traversable`.

    step_cost is a per-cell multiplier (>= 1) used to make paths prefer open
    space and known-free ground. Returns distances in metres, np.inf where
    unreachable. One sweep gives the cost to *every* candidate at once,
    which is why this beats issuing an N-way ComputePathToPose query per
    planning cycle.
    """
    dist = np.full(traversable.shape, np.inf, dtype=np.float64)
    h, w = traversable.shape
    heap: list[tuple[float, int, int]] = []
    for r, c in sources:
        if 0 <= r < h and 0 <= c < w:
            dist[r, c] = 0.0
            heap.append((0.0, r, c))
    heapq.heapify(heap)

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, geom in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not traversable[nr, nc]:
                continue
            nd = d + geom * resolution * step_cost[nr, nc]
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def nearest_where(mask: np.ndarray, r0: int, c0: int, max_radius: int) -> tuple[int, int] | None:
    """Breadth-first search outward from (r0, c0) for the closest True cell."""
    h, w = mask.shape
    if 0 <= r0 < h and 0 <= c0 < w and mask[r0, c0]:
        return r0, c0
    seen = {(r0, c0)}
    q = deque([(r0, c0, 0)])
    while q:
        r, c, depth = q.popleft()
        if depth >= max_radius:
            continue
        for dr, dc, _ in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            if mask[nr, nc]:
                return nr, nc
            q.append((nr, nc, depth + 1))
    return None


def raycast_gain(codes: np.ndarray, r0: int, c0: int, max_steps: int, num_rays: int) -> int:
    """Approximate visibility-based information gain at a viewpoint:

        I(v) = |{ x in Unknown : x potentially visible from v }|

    A virtual 360-degree laser is cast into the grid; each ray stops at the
    first OCC cell (or the map edge) and every UNK cell before that counts
    once. np.unique deduplicates cells hit by several rays.

    Note the resolution ceiling this shares with any ray-based estimate: at
    radius R the rays are R * 2*pi / num_rays apart, so beyond
    num_rays * res / (2*pi) the sampling under-counts. Keep num_rays at
    roughly 2*pi*max_steps to stay honest at full range. The bias is
    systematic across candidates, so ranking survives it either way.
    """
    h, w = codes.shape
    ang = 2.0 * np.pi * np.arange(num_rays) / num_rays
    # nav_msgs row index grows with +y, hence +sin (the provided sim_core
    # uses -sin because its own grid is stored y-flipped).
    dr = np.sin(ang)
    dc = np.cos(ang)
    steps = np.arange(1, max_steps + 1)

    rr = (r0 + dr[:, None] * steps[None, :]).astype(np.int64)
    cc = (c0 + dc[:, None] * steps[None, :]).astype(np.int64)
    inb = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rq = np.clip(rr, 0, h - 1)
    cq = np.clip(cc, 0, w - 1)
    vals = codes[rq, cq]

    blocked = (~inb) | (vals == OCC)
    any_blocked = blocked.any(axis=1)
    first = np.where(any_blocked, blocked.argmax(axis=1), max_steps)
    visible = np.arange(max_steps)[None, :] < first[:, None]

    hit = visible & inb & (vals == UNK)
    if not hit.any():
        return 0
    return int(np.unique(rq[hit] * w + cq[hit]).size)


# ------------------------------------------------------------- the planner


@dataclass
class PlannerConfig:
    """Everything tunable. Distances in metres, times in seconds."""

    # occupancy thresholds -- mirror nav2_map_server's defaults
    free_thresh: int = 25
    occ_thresh: int = 65

    # geometry
    robot_radius_m: float = 0.2
    # Extra clearance demanded only of poses the robot must *stop* at.
    # Passing through is a separate, looser test (see build_view): requiring
    # stopping clearance everywhere disconnects narrow doorways and caps
    # reachable area at ~50% of the scored denominator on the tighter maps.
    safety_margin_m: float = 0.04
    # Planning-grid resolution. The map is 0.015 m/cell; Dijkstra over the
    # full-resolution grid costs ~250 ms per sweep in Python, so we coarsen.
    # Nav2 still drives on its own 0.02 m costmap -- this grid only decides
    # *where* to go, not how to get there.
    target_cell_m: float = 0.07

    # traversal shaping
    unknown_cost_mult: float = 1.6
    inflation_radius_m: float = 0.45
    inflation_weight: float = 2.5

    # frontiers
    min_cluster_cells: int = 3
    max_candidates: int = 14
    # A long frontier arc has a centroid sitting in open space, nowhere near
    # the arc itself. Sample along each cluster instead of trusting one point.
    cluster_samples: int = 4
    sample_stride_cells: int = 6
    viewpoint_min_sep_m: float = 0.9
    # Nav2's xy_goal_tolerance is 0.25 m; a goal inside that is reported
    # reached without the robot moving, which deadlocks the whole loop.
    min_goal_distance_m: float = 0.6

    # bootstrap probes (see Planner._probes). Probes may target unknown
    # space, so they are strictly a bootstrap tool: once a real map exists,
    # a goal in unknown space is as likely to be inside a wall as in a room,
    # and driving at one wedges the robot.
    probe_min_m: float = 1.0
    probe_max_m: float = 2.5
    probe_count: int = 12
    bootstrap_free_cells: int = 400

    # escape manoeuvre
    escape_min_m: float = 0.7
    escape_max_m: float = 2.0

    # how far from the raw home estimate we may snap to find a plannable goal
    return_snap_max_m: float = 1.0

    # information gain
    sensor_range_m: float = 6.0
    num_rays: int = 720
    min_gain_cells: int = 8

    # utility weights
    w_info: float = 1.0
    w_path: float = 0.55
    w_home: float = 0.30
    w_revisit: float = 0.35
    w_turn: float = 0.15

    # revisit bookkeeping
    revisit_radius_m: float = 0.7
    blacklist_radius_m: float = 0.5

    # return-home budgeting
    speed_efficiency: float = 0.5  # fraction of max_linear_vel actually achieved
    max_linear_vel: float = 0.3
    return_safety: float = 1.6
    return_margin_s: float = 45.0


@dataclass
class Candidate:
    row: int
    col: int
    x: float
    y: float
    gain: int
    path_cost: float
    home_cost: float
    revisit: float
    turn: float
    utility: float = 0.0
    cluster_size: int = 0
    source: str = "frontier"



@dataclass
class PlanResult:
    """One planning cycle's output. Bundles the return-cost lookup with the
    candidates so a caller never has to re-run Dijkstra to ask "how far am I
    from home right now?"."""

    candidates: list[Candidate]
    robot_home_cost: float
    home_reachable: bool


@dataclass
class GridView:
    """A classified, downsampled snapshot of /map plus its frame maths."""

    codes: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    clearance_m: np.ndarray
    traversable: np.ndarray
    step_cost: np.ndarray

    def to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((y - self.origin_y) / self.resolution)),
            int(math.floor((x - self.origin_x) / self.resolution)),
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.codes.shape[0] and 0 <= col < self.codes.shape[1]


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a degenerate spread collapses to zeros so the term
    simply drops out of the utility instead of exploding."""
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class Planner:
    """Stateless per-cycle scoring plus the small amount of history that
    makes the revisit penalty and the blacklist meaningful.

    History is kept in world coordinates, never cell indices: the map grows
    and its origin shifts as slam_toolbox extends the grid, so a cached
    (row, col) silently means something different a minute later.
    """

    def __init__(self, config: PlannerConfig) -> None:
        self.cfg = config
        self.visited: list[tuple[float, float]] = []
        self.blacklist: list[tuple[float, float]] = []

    # -- history -----------------------------------------------------------

    def note_visit(self, x: float, y: float) -> None:
        if not self.visited or math.hypot(x - self.visited[-1][0], y - self.visited[-1][1]) > 0.25:
            self.visited.append((x, y))

    def note_failure(self, x: float, y: float) -> None:
        self.blacklist.append((x, y))

    def is_blacklisted(self, x: float, y: float) -> bool:
        rr = self.cfg.blacklist_radius_m
        return any(math.hypot(x - bx, y - by) < rr for bx, by in self.blacklist)

    def _revisit_score(self, x: float, y: float) -> float:
        rr = self.cfg.revisit_radius_m
        return float(sum(1 for vx, vy in self.visited if math.hypot(x - vx, y - vy) < rr))

    # -- map preparation ---------------------------------------------------

    def build_view(self, data: np.ndarray, resolution: float, origin_x: float, origin_y: float) -> GridView:
        cfg = self.cfg
        codes_fine = classify(data, cfg.free_thresh, cfg.occ_thresh)
        k = max(1, int(round(cfg.target_cell_m / resolution)))
        codes = downsample(codes_fine, k)
        res = resolution * k

        clearance = obstacle_distance(codes) * res
        # Traversability uses the bare robot radius. Anything stricter here
        # closes passages the robot physically fits through, and since the
        # coverage denominator is computed at the true radius, that is
        # unreachable area we could never score.
        traversable = (codes != OCC) & (clearance >= cfg.robot_radius_m)

        # Prefer open, known ground: unknown cells cost more, and cells close
        # to a wall cost more, so a path drifts toward the middle of a
        # corridor instead of scraping along it.
        step = np.ones(codes.shape, dtype=np.float64)
        step[codes == UNK] = cfg.unknown_cost_mult
        if cfg.inflation_radius_m > 0:
            squeeze = np.clip((cfg.inflation_radius_m - clearance) / cfg.inflation_radius_m, 0.0, 1.0)
            step += cfg.inflation_weight * squeeze
        return GridView(codes, res, origin_x, origin_y, clearance, traversable, step)

    def snap(self, view: GridView, x: float, y: float, max_radius: int = 12) -> tuple[int, int] | None:
        """Nearest traversable cell to a world point. The robot's own cell can
        read as non-traversable right after a conservative downsample, so
        callers must snap rather than trust the raw index."""
        row, col = view.to_cell(x, y)
        if not view.in_bounds(row, col):
            return None
        return nearest_where(view.traversable, row, col, max_radius)

    # -- the pipeline ------------------------------------------------------

    def _proposals(self, view: GridView, clusters, reachable_safe, cost_from_robot) -> list:
        """Frontier clusters -> deduplicated safe viewpoint cells.

        Each cluster contributes several samples spread along it rather than
        one centroid, so a long arc around the sensor horizon yields
        viewpoints at both ends instead of a single point sitting in the
        middle of already-known space. Proposals closer than
        viewpoint_min_sep_m to an accepted one are dropped -- neighbouring
        clusters otherwise burn the whole candidate budget on one spot.
        """
        cfg = self.cfg
        min_sep_cells = max(1.0, cfg.viewpoint_min_sep_m / view.resolution)
        accepted: list[tuple[tuple[int, int], int]] = []

        # Cheap prescreen so ray-casting only runs on plausible clusters:
        # big and close beats small and far.
        ordered = sorted(
            clusters,
            key=lambda cl: -len(cl) / (1.0 + self._cluster_cost(cl, cost_from_robot)),
        )
        for cl in ordered:
            if len(accepted) >= cfg.max_candidates:
                break
            n = int(np.clip(len(cl) // cfg.sample_stride_cells, 1, cfg.cluster_samples))
            idx = np.unique(np.linspace(0, len(cl) - 1, n).astype(int))
            seeds = [(int(cl[i][0]), int(cl[i][1])) for i in idx]
            centroid = cl.mean(axis=0)
            seeds.append((int(round(centroid[0])), int(round(centroid[1]))))

            for sr, sc in seeds:
                if len(accepted) >= cfg.max_candidates:
                    break
                vp = nearest_where(reachable_safe, sr, sc, max_radius=14)
                if vp is None:
                    continue
                if any(math.hypot(vp[0] - a[0], vp[1] - a[1]) < min_sep_cells for a, _ in accepted):
                    continue
                accepted.append((vp, len(cl)))
        return accepted

    def _probes(self, view: GridView, standable, cost_from_robot, robot_cell) -> list:
        """Bootstrap fallback: a ring of reachable cells around the robot.

        slam_toolbox needs a cell to be crossed by min_pass_through rays
        before it will map it at all, and rays diverge, so a *stationary*
        robot maps only a blob a few tens of centimetres across no matter how
        clean its scan is. At session start the only frontier is therefore
        inside Nav2's own goal tolerance -- Nav2 reports instant success, the
        robot never moves, and the map never grows.

        Probes break that deadlock by proposing goals 1-2.5 m out, including
        into unknown space, which Nav2 will happily plan through
        (allow_unknown: true). A probe that turns out to be inside a real
        wall simply fails and gets blacklisted.
        """
        cfg = self.cfg
        lo = cfg.probe_min_m
        hi = cfg.probe_max_m
        band = standable & np.isfinite(cost_from_robot) & (cost_from_robot >= lo) & (cost_from_robot <= hi)
        rows, cols = np.nonzero(band)
        if rows.size == 0:
            return []
        # One probe per angular sector, the cheapest of each, so the ring is
        # spread around the robot instead of clumped on one side.
        ang = np.arctan2(rows - robot_cell[0], cols - robot_cell[1])
        sector = np.floor((ang + np.pi) / (2 * np.pi) * cfg.probe_count).astype(int)
        sector = np.clip(sector, 0, cfg.probe_count - 1)
        out = []
        for s in range(cfg.probe_count):
            sel = np.nonzero(sector == s)[0]
            if sel.size == 0:
                continue
            best = sel[np.argmax(cost_from_robot[rows[sel], cols[sel]])]
            out.append(((int(rows[best]), int(cols[best])), 0))
        return out

    def _score(
        self,
        view: GridView,
        proposals,
        robot_xy,
        robot_yaw,
        home_xy,
        cost_from_robot,
        cost_to_home,
        max_steps,
        source: str = "frontier",
    ) -> list[Candidate]:
        cfg = self.cfg
        out: list[Candidate] = []
        for vp, cluster_size in proposals:
            vx, vy = view.to_world(*vp)
            if self.is_blacklisted(vx, vy):
                continue
            if math.hypot(vx - robot_xy[0], vy - robot_xy[1]) < cfg.min_goal_distance_m:
                # Below Nav2's goal tolerance the robot would be declared
                # "arrived" without moving, and we would re-pick this same
                # point forever.
                continue
            path_cost = float(cost_from_robot[vp])
            if not math.isfinite(path_cost):
                continue
            home_cost = float(cost_to_home[vp])
            if not math.isfinite(home_cost):
                home_cost = math.hypot(vx - home_xy[0], vy - home_xy[1]) * 1.3

            gain = raycast_gain(view.codes, vp[0], vp[1], max_steps, cfg.num_rays)
            if gain < cfg.min_gain_cells:
                continue

            bearing = math.atan2(vy - robot_xy[1], vx - robot_xy[0])
            turn = abs(math.atan2(math.sin(bearing - robot_yaw), math.cos(bearing - robot_yaw))) / math.pi

            out.append(
                Candidate(
                    row=vp[0],
                    col=vp[1],
                    x=vx,
                    y=vy,
                    gain=gain,
                    path_cost=path_cost,
                    home_cost=home_cost,
                    revisit=self._revisit_score(vx, vy),
                    turn=turn,
                    cluster_size=cluster_size,
                    source=source,
                )
            )
        return out

    def plan(
        self,
        view: GridView,
        robot_xy: tuple[float, float],
        robot_yaw: float,
        home_xy: tuple[float, float],
        time_pressure: float = 0.0,
    ) -> PlanResult:
        """One full cycle: Frontiers -> safe viewpoints -> scored candidates.

        Deliberately a single entry point. Each Dijkstra sweep costs ~100 ms
        of Python, so the robot->* and home->* fields are computed once here
        and everything downstream -- candidate path costs, return costs, and
        the robot's own distance home -- is read out of them.

        time_pressure in [0, 1] is the fraction of the remaining budget the
        return leg already claims. At 0 (plenty of time) the return-cost term
        nearly vanishes and the robot is free to push deep; as it approaches
        1 the term dominates and candidates near home win. A fixed w_home
        would either time out on big maps or refuse to leave the room on
        small ones.
        """
        cfg = self.cfg
        euclid_home = math.hypot(robot_xy[0] - home_xy[0], robot_xy[1] - home_xy[1])

        start = self.snap(view, *robot_xy)
        if start is None:
            return PlanResult([], euclid_home * 1.3, False)
        cost_from_robot = dijkstra(view.traversable, view.step_cost, [start], view.resolution)

        home_cell = self.snap(view, *home_xy, max_radius=20)
        home_reachable = False
        if home_cell is None:
            cost_to_home = np.full_like(cost_from_robot, np.inf)
            robot_home_cost = euclid_home * 1.3
        else:
            cost_to_home = dijkstra(view.traversable, view.step_cost, [home_cell], view.resolution)
            at_robot = float(cost_to_home[start])
            home_reachable = math.isfinite(at_robot)
            # Straight line inflated by 1.3 when the grid says "unreachable".
            # Being pessimistic about the return leg is cheap; being
            # optimistic about it is the one mistake with no recovery.
            robot_home_cost = at_robot if home_reachable else euclid_home * 1.3

        max_steps = max(1, int(round(cfg.sensor_range_m / view.resolution)))
        clear_ok = view.clearance_m >= (cfg.robot_radius_m + cfg.safety_margin_m)
        # Somewhere the footprint fits, known-free or not. Unknown cells count
        # because clearance is measured against *known* walls only, matching
        # Nav2's allow_unknown planner.
        standable = view.traversable & clear_ok
        reachable_safe = standable & (view.codes == FREE) & np.isfinite(cost_from_robot)

        clusters = cluster_cells(frontier_mask(view.codes), cfg.min_cluster_cells)
        raw: list[Candidate] = []
        if clusters:
            raw = self._score(
                view,
                self._proposals(view, clusters, reachable_safe, cost_from_robot),
                robot_xy,
                robot_yaw,
                home_xy,
                cost_from_robot,
                cost_to_home,
                max_steps,
            )

        # Probes only while the map is still the tiny blob slam_toolbox gives
        # a stationary robot. After that, "no frontier candidates" means
        # exploration is genuinely finished, not that we should go poking at
        # unknown space.
        if not raw and int((view.codes == FREE).sum()) < cfg.bootstrap_free_cells:
            raw = self._score(
                view,
                self._probes(view, standable, cost_from_robot, start),
                robot_xy,
                robot_yaw,
                home_xy,
                cost_from_robot,
                cost_to_home,
                max_steps,
                source="probe",
            )

        if not raw:
            return PlanResult([], robot_home_cost, home_reachable)

        info = _normalise(np.array([c.gain for c in raw], dtype=np.float64))
        path = _normalise(np.array([c.path_cost for c in raw], dtype=np.float64))
        home = _normalise(np.array([c.home_cost for c in raw], dtype=np.float64))
        revisit = _normalise(np.array([c.revisit for c in raw], dtype=np.float64))
        turn = _normalise(np.array([c.turn for c in raw], dtype=np.float64))

        w_home = cfg.w_home * (0.25 + 2.5 * float(np.clip(time_pressure, 0.0, 1.0)))
        for i, cand in enumerate(raw):
            cand.utility = (
                cfg.w_info * info[i]
                - cfg.w_path * path[i]
                - w_home * home[i]
                - cfg.w_revisit * revisit[i]
                - cfg.w_turn * turn[i]
            )
        raw.sort(key=lambda c: c.utility, reverse=True)
        return PlanResult(raw, robot_home_cost, home_reachable)

    @staticmethod
    def _cluster_cost(cluster: np.ndarray, cost: np.ndarray) -> float:
        vals = cost[cluster[:, 0], cluster[:, 1]]
        finite = vals[np.isfinite(vals)]
        return float(finite.min()) if finite.size else 1e6

    # -- time budgeting ----------------------------------------------------

    def travel_time(self, distance_m: float) -> float:
        cfg = self.cfg
        speed = max(0.05, cfg.max_linear_vel * cfg.speed_efficiency)
        return distance_m / speed

    def reserve_for_return(self, home_cost_m: float) -> float:
        """Sim seconds to hold back so the return leg is never a gamble."""
        cfg = self.cfg
        return self.travel_time(home_cost_m) * cfg.return_safety + cfg.return_margin_s

    def affordable(self, cand: Candidate, remaining_s: float) -> bool:
        """Can we reach this viewpoint *and still get home* in the time left?"""
        out = self.travel_time(cand.path_cost)
        back = self.reserve_for_return(cand.home_cost)
        return out + back <= remaining_s

    def escape_target(
        self, view: GridView, robot_xy: tuple[float, float]
    ) -> tuple[float, float] | None:
        """The most open known-free spot a short way off, for backing out of
        a wedge.

        Nav2's own spin/back-up recoveries report "Collision Ahead" and give
        up once the footprint is already touching a wall, so the escape has
        to be a normal navigation goal into demonstrably open space rather
        than a blind manoeuvre.
        """
        cfg = self.cfg
        start = self.snap(view, *robot_xy, max_radius=20)
        if start is None:
            return None
        cost = dijkstra(view.traversable, view.step_cost, [start], view.resolution)
        band = (
            (view.codes == FREE)
            & np.isfinite(cost)
            & (cost >= cfg.escape_min_m)
            & (cost <= cfg.escape_max_m)
        )
        if not band.any():
            return None
        clearance = np.where(band, view.clearance_m, -np.inf)
        idx = int(np.argmax(clearance))
        row, col = divmod(idx, view.codes.shape[1])
        return view.to_world(row, col)

    def return_target(
        self, view: GridView, home_xy: tuple[float, float], robot_xy: tuple[float, float]
    ) -> tuple[tuple[float, float], float] | None:
        """The closest cell to home that the robot can actually stand on and
        reach, plus how far that is from the raw home estimate.

        Home is the odom origin re-derived through map->odom, and that
        estimate carries all of SLAM's accumulated correction: in practice it
        can land a couple of centimetres from a mapped wall, i.e. inside
        Nav2's inscribed radius, where the planner refuses to plan at all and
        every return attempt aborts. Snapping to the nearest genuinely
        occupiable cell keeps the goal plannable while staying as close to
        home as the map allows.

        Clearance here is the bare robot radius, not the padded exploration
        margin -- the robot demonstrably fit at home, since it started there.
        """
        cfg = self.cfg
        start = self.snap(view, *robot_xy, max_radius=20)
        if start is None:
            return None
        cost = dijkstra(view.traversable, view.step_cost, [start], view.resolution)
        occupiable = (
            (view.codes == FREE)
            & (view.clearance_m >= cfg.robot_radius_m)
            & np.isfinite(cost)
        )
        hr, hc = view.to_cell(*home_xy)
        if not view.in_bounds(hr, hc):
            return None
        radius_cells = max(2, int(round(cfg.return_snap_max_m / view.resolution)))
        cell = nearest_where(occupiable, hr, hc, radius_cells)
        if cell is None:
            return None
        wx, wy = view.to_world(*cell)
        return (wx, wy), math.hypot(wx - home_xy[0], wy - home_xy[1])

    def return_targets(
        self,
        view: GridView,
        home_xy: tuple[float, float],
        robot_xy: tuple[float, float],
        count: int = 5,
    ) -> list[tuple[tuple[float, float], float]]:
        """Occupiable, reachable cells near home, nearest-to-home first.

        A single target is not enough. This grid is built from the static map
        alone, while Nav2's global costmap also marks live laser returns, so a
        cell this planner calls occupiable can still be refused. Handing back a
        ranked list lets a refusal fall through to the next candidate instead
        of spending every return attempt on the same unreachable cell.
        """
        cfg = self.cfg
        start = self.snap(view, *robot_xy, max_radius=20)
        if start is None:
            return []
        cost = dijkstra(view.traversable, view.step_cost, [start], view.resolution)
        occupiable = (
            (view.codes == FREE)
            & (view.clearance_m >= cfg.robot_radius_m)
            & np.isfinite(cost)
        )
        rows, cols = np.nonzero(occupiable)
        if rows.size == 0:
            return []
        wx = view.origin_x + (cols + 0.5) * view.resolution
        wy = view.origin_y + (rows + 0.5) * view.resolution
        d = np.hypot(wx - home_xy[0], wy - home_xy[1])
        keep = d <= cfg.return_snap_max_m
        if not keep.any():
            keep = d <= (cfg.return_snap_max_m * 2.0)
        if not keep.any():
            return []
        order = np.argsort(d[keep])
        cand_x, cand_y, cand_d = wx[keep][order], wy[keep][order], d[keep][order]

        out: list[tuple[tuple[float, float], float]] = []
        for x, y, dist in zip(cand_x.tolist(), cand_y.tolist(), cand_d.tolist()):
            if any(math.hypot(x - px, y - py) < 0.25 for (px, py), _ in out):
                continue
            out.append(((x, y), dist))
            if len(out) >= count:
                break
        return out
