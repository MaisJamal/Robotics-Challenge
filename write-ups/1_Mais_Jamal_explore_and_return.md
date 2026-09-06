# Explore and Return: RA-UFE

Author: Mais Jamal

## Approach

Return-Aware Utility Frontier Exploration (RA-UFE) follows **map → frontier clusters → safe viewpoints → utility ranking → navigation → re-evaluation**. `ra_ufe.py` implements the NumPy planner; `explorer_node.py` handles ROS, TF, Nav2 actions, recovery, and finishing. The provided simulator and SLAM are unchanged.

The planner conservatively downsamples `/map`, samples multiple viewpoints per frontier, and estimates information gain with a virtual 360° laser. Dijkstra estimates travel and return costs.

I prioritize coverage while reserving return time. Measured travel speed informs the reserve, and elapsed-time pressure increasingly favors homeward goals. Exploration also has a time cap and an estimated-coverage target. This coverage estimate is a proxy, not the evaluator's ground-truth percentage.

## Heuristic function

Following the utility-based frontier strategy, I select the highest-scoring feasible viewpoint rather than simply the nearest frontier. Every candidate viewpoint (from frontier) is scored as follows:

```text
U = w_I * Î - w_P * Ĉ - w_home(t) * Ĥ - w_R * R̂ - w_D * D̂

```

- **I:** expected information gain; predicted unique unknown cells visible within the 6 m laser range.
- **C:** path cost to frontier; weighted path cost from the robot, accounting for detours and obstacle proximity.
- **H:** estimated future path cost from viewpoint (frontier) to home.
- **R:** revisit/redundancy penalty; nearby visit count plus a decaying failure penalty; recent failures also temporarily exclude goals.
- **D:** direction-change penalty; heading change toward the viewpoint, divided by π.

Current weights are set to : 
- w_I = 1.0
- w_P = 0.55
- w_R = 0.35
- w_D = 0.35
```text
U = 1.0·Î − 0.55·Ĉ − w_home(t)·Ĥ − 0.35·R̂ − 0.35·D̂

```

Gain and travel use `Î = I/(I+150)` and `Ĉ = C/(C+4)`. Home, revisit, and turning terms use min-max normalization across candidates; constant terms contribute zero. Unlike the PDF's separate failure term, failure memory is included in R.

```text
pressure = clip(return_reserve / remaining_time + 1.5·elapsed_time / time_limit, 0, 1)
w_home(t) = 0.30·(0.25 + 2.5·pressure)
```

This increases the home penalty as time passes. Separate affordability checks reserve enough estimated time for the outward and return journeys. The heuristic balances useful observation against travel and return burden, but does not guarantee optimal ordering or completion of one room before moving elsewhere.

## Issues addressed during development

- **Startup failures:** enforced simulation time, waited for Nav2 readiness, and avoided blacklisting goals when the action server was unavailable.
- **Sparse-map deadlock:** added bounded bootstrap probes, minimum goal displacement, and zero return cost when odometry confirms the robot is already home.
- **Poor viewpoints:** sampled and deduplicated frontier viewpoints; filtered too-close and temporarily blocked positions before selecting alternatives. Traversal and stopping clearance are separate.
- **Incorrect route costs:** prevented diagonal corner cutting, made edge costs symmetric, and removed straight-line substitutes for unverified return routes. Full-resolution checks recover routes lost through downsampling for both the robot and candidate viewpoints.
- **Premature completion:** rejected proposals no longer imply exhausted exploration. Return uncertainty is classified as an endpoint issue, unknown gap, or obstacle/clearance blockage and rechecked against fresh maps before returning.
- **Unstable selection:** replaced permanent blacklists with timed backoff and decaying penalties, stabilized gain/travel scaling, and compared current-cycle goal alternatives. Exact active-goal re-scoring remains unfinished.
- **Overcoming stuck episodes:** I added reverse-and-turn maneuvers with alternating turn direction to free the robot when ordinary navigation stalled. These helped it escape some wedges and resume exploration or return home, although recovery is not reliable in every case.
- **Action and recovery handling:** ignored stale results, canceled late-accepted obsolete goals, enabled return watchdogs, and made unstick wait for navigation to stop and publish zero velocity on exit. Session deadlines override active navigation and recovery.
- **Home and finish loops:** recomputed home using current TF, tried alternative approach targets, spaced retries, relaxed the arrival threshold, and distinguished a finalized failed grade from a failed finish-service request.
- **Wall contact:** experimented with costmap inflation and a 0.22 m navigation radius. These changes trade clearance against doorway accessibility; they have not eliminated wedging.

Offline regression tests cover these fixes; simulation is still necessary to assess navigation performance.

## Results and remaining limitations

Maps **1, 4 and 5** have always successful exploration and return with different seeds and time-scales.

Maps **2 and 3 are the largest** and were the most demanding. In my testing, map 3 produced two failures that I considered close to success; they remain failures under the strict scoring rules, not successful runs.

Map 2 repeatedly showed SLAM/map-alignment problems: sudden estimated-pose jumps, duplicated walls, and scan/map disagreement. The underlying cause is not yet confirmed. One map 2 run achieved **99.76% coverage** but finished **0.539 m from true home**, outside the **0.30 m** requirement, despite the explorer estimating **0.18 m**.

Remaining weaknesses include odometry-based home error, sparse-map verification deadlocks, long detours, and open-loop recovery. Nav2 failures occurring before the watchdog timeout can still bypass the unstick trigger.

## With more time

I would tune the utility-function weights and scaling, particularly the balance between information gain, travel, turning, and return pressure. I would investigate maps 2 and 3 further using repeated seeds and recorded scans/TF, improve final home localization, and replace repetitive recovery with clearance-aware maneuvers.
