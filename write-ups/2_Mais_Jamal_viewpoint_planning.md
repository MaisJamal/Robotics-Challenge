# Write-Up: Minimum-Stop Viewpoint Planning for Accurate Scanning

Author: Mais Jamal

## Approach

This submission uses a deterministic greedy baseline, refined through candidate sampling and
coverage-target experiments. I prioritize measured wall coverage and a drivable route, while
using stop pruning and tour heuristics to control the cost. I do not claim an optimal solution.

**Starting point: the sensor model.** My first thought was that the plan should get the most out
of the `min_quality` idea — I want every scan to be *effective*, not merely to touch a wall cell.
Since the quality falloff is given (`quality_at_range` in `sim/visibility.py`), I solved it for
the range at which a cell still counts:

```
q(r) = 1 - (r / R_max)^2  >=  min_quality
  =>  r_eff = R_max * sqrt(1 - min_quality)
```

With the fixed evaluation settings (`R_max = 8 m`, `min_quality = 0.5`) this gives
**r_eff ≈ 5.66 m**. Anything beyond that range is invisible to the score, so `r_eff` is the
length scale the whole plan is built around.

**Pipeline.** From there the planner runs in four stages:

1. **Candidate generation.** Lay a 0.75 m lattice over free space, add 0.25 m samples within
   a 1 m square neighborhood of walls to give doorways and corners more alternatives, and keep
   only the poses where the robot's footprint actually fits, using `traversable_mask` — which
   applies the same clearance test as the scorer's `is_stop_valid`. Every candidate is therefore
   valid by construction, and the plan can never lose points to a colliding or non-traversable
   stop.
2. **Connected-component filtering.** Group the surviving candidates into connected free-space
   components and plan inside a single one. I deliberately do **not** take the *largest*
   component: on some of these maps the (mis-read) outdoor area is larger than the indoor area,
   so "largest" would pick exactly the wrong region. Instead I score each component by the wall
   coverage its candidates can *jointly* reach — the union of their scan sets — and keep the
   best-scoring one.
3. **Stop selection (set cover).** For each candidate I ray-cast a full 360° scan
   (`scan_from_stop`) and record which observable wall cells it sees at `>= min_quality`.
   Finding the fewest stops that cover the most wall is exactly the **maximum-coverage /
   set-cover** problem, which is NP-hard, so for a baseline I solve it with the classic
   **greedy** algorithm: repeatedly take the candidate that adds the most still-uncovered wall,
   until 95% of the selected component's candidate coverage union is covered. Then remove
   stops while preserving that target. There is no marginal-gain floor or fixed stop budget.
   This is a heuristic, not a proof of the minimum number of stops.
4. **Tour ordering.** Once the stops are chosen, shorten the route that visits them:
   nearest-neighbor from every possible start, then **2-opt** refinement, keeping the best
   result. Both run over **geodesic (free-space) distances** from `multi_target_shortest_paths`,
   not straight lines — the scorer reports a routed distance, so optimizing against
   straight-line distance would be optimizing the wrong objective and would mis-order any tour
   whose stops sit in different rooms.

**Determinism.** There is no randomness anywhere in the pipeline. The candidate lattice is
generated in row-major order, and every tie — in greedy selection, in component choice, in
nearest-neighbor — is resolved by index. 2-opt keeps the first best reversal in its fixed
index scan order, and equal-length tours keep the first start tested. The same map and sensor
therefore produce the same stops in the same order.

**What I observed while tuning.** The first runs gave a genuinely small stop count but low
coverage (~20–30%). I then swept the parameters — reducing the spacing between candidate stops
well below `r_eff` so that greedy has real choice about where to stand, changing how the
component is selected, and raising the stop budget — and found that coverage across all maps
did not exceed roughly 70% in those experiments. This is an observed limit of the tested
candidates, not a proven geometric ceiling; disconnected regions, sampling and occlusion
all affect it. The current planner instead targets most of the coverage attainable by its
candidates in a single drivable region.

## Design Decisions & Tradeoffs

The rubric ranks **coverage > stop count > tour length**, so coverage is where the budget should
go. I chose a target of 95% of the selected component's candidate coverage union after comparing
90%, 95% and 99.5%. This deliberately accepts more stops than the earlier conservative baseline.

**Candidate spacing (0.75 m, refined to 0.25 m near walls).** Well under `r_eff ≈ 5.66 m`. Spacing candidates at `r_eff` — one
stop per sensor footprint — was my first instinct, but it makes the lattice too coarse for
greedy to have any real choice, and a candidate that happens to land in a doorway or behind a
pillar then has no nearby alternative. Denser candidates cost planning time (a scan is ~5 ms and
the whole lattice is scanned once) but buy a much better greedy trajectory. This is the
coverage-vs-runtime dial, and runtime is not scored, so I biased toward density.

**Coverage target instead of a marginal-gain floor.** Each scan credits at most 720 distinct
wall cells, and later scans often contribute only small additions. The earlier gain threshold
stopped selection even when useful coverage remained. I removed that threshold and the 40-stop
cap. The internal target is now `ceil(0.95 * candidate_union_size)`; the official score still
divides by all observable wall cells on the map. Pruning removes stops only when the remaining
scans still meet the internal target.

**Comparing coverage/stop operating points.** I reused the same candidate scan sets and selected
component for each target. Each table entry is **official wall coverage / selected stops**,
after pruning. The column headings are internal candidate-union targets, not official scores.

| Map | 90% target | 95% target (chosen) | 99.5% target |
|---|---:|---:|---:|
| 1 | 39.68% / 35 | 41.82% / 55 | 43.75% / 137 |
| 2 | 37.61% / 146 | 39.69% / 252 | 41.57% / 652 |
| 3 | 39.79% / 91 | 41.99% / 152 | 43.98% / 408 |
| 4 | 32.86% / 14 | 34.66% / 21 | 36.27% / 46 |
| 5 | 40.70% / 40 | 42.89% / 63 | 44.92% / 140 |

I chose 95% to prioritize coverage while leaving the final expensive gains. Moving from 90%
to 95% adds about 1.8–2.2 percentage points of official coverage. Moving from 95% to 99.5%
adds only about 1.6–2.0 more points but requires over twice as many stops on every map.
For example, map 2 needs 400 additional stops for another 1.88 percentage points. A strict
coverage-only choice would favor 99.5%; 95% is my explicit trade-off between the three axes,
not a claim that it maximizes the primary score. Only the chosen 95% plans were fully routed
in this comparison, so I do not claim measured travel or runtime improvements for the other targets.

**Offline planning and runtime.** I treat this challenge as offline planning: the complete map
and sensor model are available before the robot visits the stops. I therefore did not prioritize
runtime optimization; my focus was wall coverage, stop count and drivable tour length, the three
ranked evaluation criteria. Maps 2 and 3 can take several minutes to plan. In the diagnostic
runs, which also included the target comparisons and repeated selection checks, planning took
about 10.5 minutes for map 2 and 5.5 minutes for map 3. These are observations from that test
environment, not isolated runtime benchmarks; hardware and concurrent evaluations affect timing.
More selected stops make the geodesic distance calculations and multi-start tour optimization
more expensive. Choosing a lower coverage target can reduce that work, but comparing targets
does not itself speed up the planner. A deployment requiring frequent replanning would need
additional runtime optimization.

**Restricting the plan to one connected component.** This is the decision with the largest cost,
and it is a deliberate trade of coverage for *honesty*. These maps are 16-bit PGMs; PIL opens
them as mode `"I"` and `.convert("L")` in `map_io` clips rather than scales, so the "unknown"
gray saturates to 255 and loads as **FREE**. The consequence is that the unmapped area outside
the building reads as drivable, and free space fragments into many disconnected components
separated by walls and clearance restrictions. The number of components represented by
candidates also depends on sampling density. If stops are allowed to spread across
components, the scorer finds no route between them and silently substitutes a straight-line
distance *through the walls* — the reported `tour_length_m` would be a number the robot could
never actually drive. Confining the plan to one component costs real coverage, but it makes
metric 3 honest. I would rather report a lower coverage that is genuinely achievable than a
better-looking tour that is fictional.

**Geodesic distances instead of Euclidean.** This costs `O(N)` multi-target Dijkstra searches
(one per stop, not `O(N²)` pairwise), which is a meaningful share of planning time on the larger
maps. It is worth it: it optimizes the tour against the metric it is actually graded on.

**2-opt on an *open* path.** The robot does not drive back to its first stop, and the scorer only
sums consecutive pairs, so the tour is a path, not a cycle. Each candidate reversal is scored by
the `O(1)` change in the two edges it breaks, rather than by recomputing the whole tour length —
at these stop counts that is the difference between a pass costing thousands of operations and
one costing millions.

**Why the official score remains below the internal target.** The scorer counts observable wall
cells across the whole map, including outer faces adjacent to areas loaded as free. Some of
those faces cannot be seen from the selected connected region. Buried wall cells are already
excluded from its denominator. Footprint clearance, occlusion, candidate spacing and discrete
ray sampling can leave additional counted cells unseen. The candidate union measures what the
current samples can cover; denser or better-placed candidates may expand it. Thus reaching 95%
of that union does not mean reaching 95% official wall coverage or proving that the remainder
is geometrically impossible.

## Conclusions

Denser candidates and a target based on their attainable coverage improved the primary score
on every sample map. I kept the single-component restriction and geodesic routing: these make
the selected stops mutually reachable under the evaluator's motion model. Further tour
optimization is secondary to improving coverage and reducing stops at comparable coverage.
The resulting plans measure more wall than the earlier conservative plans, at the cost of
more stops, longer travel and higher planning time.

### Performance

The table compares the original saved reports from September 4 with the chosen 95% plans
from the September 7 diagnostic runs. It includes both candidate-placement and stop-selection
improvements. Coverage is the official global-wall score, not the internal candidate-union
percentage. Each map number links to the corresponding new report.

| Map | Original coverage | Current coverage | Stops (original → current) | Travel in m (original → current) |
|---|---:|---:|---:|---:|
| [1](../results/viewpoint_planning/20260907_044818_map1/coverage_report.json) | 27.47% | 41.82% | 10 → 55 | 25.51 → 54.86 |
| [2](../results/viewpoint_planning/20260907_045816_map2/coverage_report.json) | 25.74% | 39.69% | 37 → 252 | 115.05 → 203.23 |
| [3](../results/viewpoint_planning/20260907_045316_map3/coverage_report.json) | 28.39% | 41.99% | 26 → 152 | 84.70 → 147.96 |
| [4](../results/viewpoint_planning/20260907_044747_map4/coverage_report.json) | 22.44% | 34.66% | 5 → 21 | 7.44 → 16.44 |
| [5](../results/viewpoint_planning/20260907_044809_map5/coverage_report.json) | 27.18% | 42.89% | 11 → 63 | 26.08 → 48.12 |

All five new plans had zero invalid stops. The diagnostic checks also confirmed that every
consecutive stop pair had a routed path, so none of these tour lengths used the scorer's
straight-line fallback. The report PNG and diagnostic log are saved beside each JSON.
The runtime figures include diagnostic work as explained above; they are not isolated
benchmarks. I have not established that these tours are globally shortest or that the plans
dominate another solver at equal coverage.

### What I'd Do With More Time

- Refine candidates around still-uncovered walls and narrow openings, then measure whether
  this expands the candidate union within the selected component.
- Compare greedy selection and pruning with a set-cover solver or stop exchanges at the
  same coverage target, to test whether fewer scans can deliver comparable coverage.
- Investigate the 16-bit map-loading behavior separately and compare against correctly
  interpreted maps, keeping the supplied evaluator's scores distinguishable.
- Profile geodesic distance calculation and multi-start 2-opt before optimizing runtime.
  Compare route improvements on identical stop sets so coverage and stop count stay comparable.

### Known Limitations / Where I Expect This to Break

- A single connected component cannot cover walls visible only from other disconnected
  regions. Choosing the component by candidate-union size does not guarantee the best
  coverage/stop/travel trade-off across all components, or suitability for a fixed robot start.
- Regular samples can miss useful poses in narrow spaces or behind occlusions. The 95% target
  describes the current samples' union, not all geometrically achievable scans.
- Greedy selection, individual-stop pruning and 2-opt are local heuristics. Pruning stops
  when no individual removal preserves the target; it can miss better combinations of stops.
- The sensor uses 720 discrete rays with distance-based quality. It does not model incidence
  angle, localization error or measurement noise. Validity and connectivity checks establish
  feasibility under the evaluator, not all constraints of a physical robot.
- Planning cost grows with candidate count, map size and selected stops. The current approach
  is intended for offline use and is not demonstrated to meet a real-time replanning deadline.

## Reproducibility

The planner uses no randomness. Repeated stop selection was checked on all five maps, and
full-plan repeatability was checked on map 4. The randomized synthetic unit-test fixture uses
seed `17`; this is test data generation, not planner randomness. Six unit tests cover the
candidate-union denominator, small gains and more than 40 stops, redundant-stop removal,
empty unions, duplicate cells, target preservation and deterministic selection.

From the repository root, run the selection tests with:

```bash
PYTHONPATH=viewpoint_planning python3 -m unittest discover -s viewpoint_planning/tests -v
```

For a normal evaluation, use the challenge runner or, with its dependencies installed:

```bash
python3 viewpoint_planning/eval.py --map maps/1/room.yaml --results-dir viewpoint_planning/results
```

## Modified/Added Files or Packages (optional)

- `viewpoint_planning/candidate_solution/solution.py`: denser candidates near walls,
  component selection, greedy coverage targeting and pruning, and geodesic tour ordering.
- `viewpoint_planning/tests/test_stop_selection.py`: regression tests for stop selection.
- `results/viewpoint_planning/20260907_*_map*/`: measured reports, visualizations and
  diagnostic logs supporting the tables. Earlier result folders remain as historical runs.
- This write-up records the approach, comparisons and limitations. These planner changes
  use the existing NumPy dependency and supplied simulator helpers; no new package is required.
