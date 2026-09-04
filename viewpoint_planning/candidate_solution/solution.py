from __future__ import annotations
"""Your task: implement plan_viewpoints().

You are given the ground-truth occupancy grid (this is the ONLY input you get —
there is no live sensor loop in this challenge, the map is already known) and
the sensor model the accurate scanner uses. Return a list of (x, y) world-frame
stop poses, IN VISIT ORDER, such that a 360-degree rotate-and-scan from each
stop, together, observes as much of the wall/boundary as possible, using as
few stops and as little travel as you can.

See README.md for the full brief, scoring rubric, and how to run
`eval.py` against your solution.
"""
"""Baseline viewpoint planner: greedy set-cover + nearest-neighbor/2-opt tour.

  1. We construct a regular grid of candidate stops over free space, keeping only 
     poses where the robot's footprint actually fits. Depending on effective radius
     of scanning and the traversable mask of the map.
  2. We define the connected groups of the candidate poses and evaluate them against
     the percentage of seen walls' cells by the function (`reachable_wall`) with 
     ray-casting a full 360-degree scan from each candidate (`scan_from_stop`) and
     record which observable wall cells it sees at >= `sensor.min_quality`.
  3. We greedily take the candidate covering the most still-uncovered wall, until
     coverage plateaus (marginal gain below a floor) or the target is hit. Using 
     greedy algorithm to solve the set-cover problem.
  4. We order the chosen stops as an open path: nearest-neighbor from every possible
     start, then 2-opt, both over *geodesic* (free-space) distances
     rather than straight lines.

Fully deterministic: no randomness anywhere, and every tie is broken by
candidate index, so the same map and sensor always give the same stops in the
same order.
"""

import numpy as np

from sim.map_io import OccupancyGrid
from sim.pathing import multi_target_shortest_paths, traversable_mask
from sim.visibility import (
    SensorModel,
    observable_wall_cells,
    quality_at_range,
    scan_from_stop,
)

# This value was taken from eval.py
# if it was not fixed in eval.py, I would make it a global variable , in 
# a global one config file, only in one place in all the project , to avoid
# duplicates and errors.
ROBOT_RADIUS_M = 0.2  

# Spacing of the candidate lattice. Well under the sensor's effective radius
# (~5.7 m at max_range 8 m / min_quality 0.5) so the greedy step has real
# choice about where to stand; scans cost ~5 ms each, so a few hundred
# candidates is affordable.
CANDIDATE_SPACING_M = 0.75

# Stop adding stops once we're this close to covering everything reachable...
COVERAGE_TARGET = 0.995

# ...or once the best remaining candidate adds too little to be worth a stop.
# This is measured against `sensor.num_rays`, NOT against the wall-cell total:
# every ray terminates at exactly one wall cell, so a single stop can credit at
# most `num_rays` (720) cells no matter how much wall is in view. On a map with
# 55k observable wall cells that ceiling is 1.3% of the map, so a threshold
# expressed as a fraction of total wall would reject every possible stop.
MIN_GAIN_RAY_FRAC = 0.25

# The primary metric is coverage and each stop can only ever add ~720 cells, so
# the stop budget is the real dial between metric 1 and metrics 2/3.
MAX_STOPS = 40

# Keep every stop inside a single connected region of free space.
#
# These maps load with no UNKNOWN cells at all: the .pgm files are 16-bit, PIL
# opens them as mode "I", and .convert("L") in map_io clips rather than scales,
# so the 52687 "unknown" gray saturates to 255 and reads as FREE. The result is
# that the unmapped area outside the building is drivable, and free space
# splits into many components (9 on map 1, 72 on map 2) separated by exterior
# walls. Stops spread across components produce a tour the robot cannot drive -
# the scorer finds no route and silently substitutes straight-line distance
# through the walls. Confining the plan to one component costs some coverage
# but makes `tour_length_m` an honest number.
RESTRICT_TO_ONE_COMPONENT = True


def quality_for_range(range_m, sensor: SensorModel):
    # Function wrapper from sim.visibility
    return quality_at_range(range_m, sensor.max_range_m)




def _candidate_poses(grid: OccupancyGrid, traversable: np.ndarray,
                     spacing_m: float) -> list[tuple[float, float]]:
    """Regular lattice of world-frame poses over traversable free space.

    `traversable_mask` applies the same footprint-clearance test as
    `is_stop_valid`, vectorized, so every pose returned here is guaranteed to
    pass the scorer's validity check.
    """
    step = max(1, int(round(spacing_m / grid.resolution)))
    lattice = np.zeros_like(traversable)
    lattice[::step, ::step] = True
    rc = np.argwhere(traversable & lattice)  # row-major => deterministic order
    return [grid.pixel_to_world(int(r), int(c)) for r, c in rc]


def _coverage_sets(grid: OccupancyGrid, poses: list[tuple[float, float]],
                   sensor: SensorModel, wall_index: np.ndarray) -> list[np.ndarray]:
    """For each pose, the indices of observable wall cells it scans at or above
    `sensor.min_quality`.

    Cells are filtered against `wall_index` (-1 for anything not in the
    scorer's coverage denominator): a ray can terminate on a wall cell whose
    only free-side neighbors are UNKNOWN, and the scorer gives no credit for
    those, so neither do we.
    """
    sets = []
    for xy in poses:
        seen = scan_from_stop(grid, xy, sensor)
        idx = [wall_index[cell] for cell, q in seen.items() if q >= sensor.min_quality]
        sets.append(np.array([i for i in idx if i >= 0], dtype=np.int32))
    return sets


def _greedy_set_cover(coverage: list[np.ndarray], num_walls: int, num_rays: int) -> list[int]:
    """Classic greedy max-coverage: repeatedly take the candidate adding the
    most uncovered cells. Ties go to the lowest candidate index, so the result
    is reproducible."""
    covered = np.zeros(num_walls, dtype=bool)
    min_gain = max(1, int(MIN_GAIN_RAY_FRAC * num_rays))
    chosen: list[int] = []

    while len(chosen) < MAX_STOPS:
        best_i, best_gain = -1, 0
        for i, cells in enumerate(coverage):
            if i in chosen or cells.size == 0:
                continue
            gain = int(np.count_nonzero(~covered[cells]))
            if gain > best_gain:  # strict: first index wins a tie
                best_i, best_gain = i, gain
        if best_i < 0 or best_gain < min_gain:
            break
        covered[coverage[best_i]] = True
        chosen.append(best_i)
        if covered.sum() / num_walls >= COVERAGE_TARGET:
            break
    return chosen


def _connected_groups(grid: OccupancyGrid, poses: list[tuple[float, float]],
                     traversable: np.ndarray) -> list[list[int]]:
    """Partition candidate poses into free-space connected components.

    Rather than implement a labeling pass, this reuses the provided
    multi-target Dijkstra: a search from one seed reaches exactly its own
    component, and reports `path is None` for every goal outside it. Each
    search therefore explores only its own component, so labeling all of them
    costs about one full traversal of free space in total.
    """
    remaining = list(range(len(poses)))
    groups: list[list[int]] = []
    while remaining:
        seed = remaining[0]
        results = multi_target_shortest_paths(
            grid, poses[seed], [poses[i] for i in remaining], traversable)
        reached = [i for i, (_d, path) in zip(remaining, results) if path is not None]
        if seed not in reached:
            reached.append(seed)
        groups.append(sorted(reached))
        left = set(reached)
        remaining = [i for i in remaining if i not in left]
    return groups


def _distance_matrix(grid: OccupancyGrid, stops: list[tuple[float, float]],
                     traversable: np.ndarray) -> np.ndarray:
    """Pairwise 8-connected free-space distances, one multi-target Dijkstra per
    stop (O(N) searches, not O(N^2)).

    Straight-line distance would mis-order any tour whose stops sit in
    different rooms — two stops a metre apart through a wall are a long drive
    — and the scorer reports the routed distance, so the tour is optimized
    against the metric it's actually graded on.
    """
    n = len(stops)
    dist = np.zeros((n, n))
    for i, start in enumerate(stops):
        for j, (length_m, _path) in enumerate(multi_target_shortest_paths(
                grid, start, stops, traversable)):
            dist[i, j] = length_m
    return dist


def _tour_length(order: list[int], dist: np.ndarray) -> float:
    return float(sum(dist[order[k], order[k + 1]] for k in range(len(order) - 1)))


def _nearest_neighbor_tour(dist: np.ndarray, start: int) -> list[int]:
    n = dist.shape[0]
    unvisited = set(range(n)) - {start}
    order = [start]
    while unvisited:
        last = order[-1]
        order.append(min(unvisited, key=lambda j: (dist[last, j], j)))
        unvisited.discard(order[-1])
    return order


def _two_opt(order: list[int], dist: np.ndarray, max_passes: int = 50) -> list[int]:
    """2-opt on an *open* path (the robot doesn't drive back to its first stop,
    and the scorer only sums consecutive pairs).

    Each candidate reversal is scored by the O(1) change in the two edges it
    breaks rather than by recomputing the whole tour: at the stop counts greedy
    produces that's the difference between a pass costing thousands of
    operations and one costing millions. Best-improvement with a fixed i/j scan
    order, so the result is deterministic.
    """
    order = list(order)
    n = len(order)
    if n < 4:
        return order

    for _ in range(max_passes):
        best_delta, best_ij = 1e-9, None
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reversing order[i:j+1] rewires at most the edge entering i
                # and the edge leaving j; a reversal spanning the whole open
                # path rewires neither, so it can never help.
                delta = 0.0
                if i > 0:
                    delta += dist[order[i - 1], order[i]] - dist[order[i - 1], order[j]]
                if j < n - 1:
                    delta += dist[order[j], order[j + 1]] - dist[order[i], order[j + 1]]
                if delta > best_delta:
                    best_delta, best_ij = delta, (i, j)
        if best_ij is None:
            break
        i, j = best_ij
        order[i:j + 1] = order[i:j + 1][::-1]
    return order


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    """Replace this. The placeholder below is intentionally bad — a single
    stop at the map's centroid — so you can see the scorer and visualization
    working end-to-end before you touch the algorithm.
    """
    free_cells = grid.free_cells()
    center_row, center_col = free_cells.mean(axis=0)
    x, y = grid.pixel_to_world(int(center_row/2), int(center_col))
    effective_radius_m =  sensor.max_range_m * max(0.0, 1.0 - sensor.min_quality) ** 0.5

    print (x,y,effective_radius_m )
    print(quality_for_range(effective_radius_m,sensor))
    print(quality_for_range(sensor.max_range_m/2,sensor))
    x_2, y_2 = grid.pixel_to_world(int(center_row/2), int(center_col+200))
    x_3, y_3 = grid.pixel_to_world(int(center_row/2), int(center_col+270))
    x_4, y_4 = grid.pixel_to_world(int(center_row/2+50), int(center_col-250))
    x_5, y_5 = grid.pixel_to_world(int(center_row/2+80), int(center_col-300))


    traversable = traversable_mask(grid, ROBOT_RADIUS_M)
    # print(traversable)
    walls = observable_wall_cells(grid)
    # print("walls:")
    # print(walls)
    wall_rc = np.argwhere(walls)
    num_walls = len(wall_rc)
    # print(num_walls)
    wall_index = np.full(grid.data.shape, -1, dtype=np.int32)
    # print(wall_index)
    wall_index[walls] = np.arange(num_walls)
    # print("wall_index 1")
    # print(wall_index[1])
    poses = _candidate_poses(grid, traversable, CANDIDATE_SPACING_M)
    # poses = _candidate_poses(grid, traversable, effective_radius_m/10)

    if not poses or num_walls == 0:
        return [(x, y),(x_2,y_2),(x_3,y_3),(x_4,y_4),(x_5,y_5)]
    #     # Degenerate map: fall back to the free-space centroid so we still
    #     # return something valid rather than an empty plan.
    #     free = grid.free_cells()
    #     r, c = free.mean(axis=0)
    #     return [grid.pixel_to_world(int(r), int(c))]
    # """
    coverage = _coverage_sets(grid, poses, sensor, wall_index)
    print("len stops at grid & traversable", len(poses))
    print("No of coverage sets", len(coverage))

    if RESTRICT_TO_ONE_COMPONENT:
        # Score each component by how much wall ALL of its candidates could
        # jointly reach, and plan inside the most promising one. Ties break on
        # the lowest candidate index, keeping the choice reproducible.
        groups = _connected_groups(grid, poses, traversable)
        print("number of groups:" , len(groups))
        def _reachable_wall(group: list[int]) -> tuple[int, int]:
            sets = [coverage[i] for i in group if coverage[i].size]
            union = np.unique(np.concatenate(sets)).size if sets else 0
            return union, -group[0]
        group = max(groups, key=_reachable_wall)
        print("No of points inside the group with max coverage(seen wall percent)",len(group))
        sub = [coverage[i] for i in group]
        # chosen = group
        chosen = [group[k] for k in _greedy_set_cover(sub, num_walls, sensor.num_rays)]
    else:
        chosen = _greedy_set_cover(coverage, num_walls, sensor.num_rays)

    if not chosen:
        return [(x, y),(x_2,y_2),(x_3,y_3),(x_4,y_4),(x_5,y_5)]
        # chosen = [int(np.argmax([c.size for c in coverage]))]


    stops = [poses[i] for i in chosen]
    # """
    # stops = poses
    print("len stops ", len(stops))
    if len(stops) <= 2:
            return stops
    
    dist = _distance_matrix(grid, stops, traversable)
    best_order, best_len = None, float("inf")
    for start in range(len(stops)):
        order = _two_opt(_nearest_neighbor_tour(dist, start), dist)
        length = _tour_length(order, dist)
        if length < best_len:  # strict: lowest start index wins a tie
            best_order, best_len = order, length

    return [stops[i] for i in best_order]

