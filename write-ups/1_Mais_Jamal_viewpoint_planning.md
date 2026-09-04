# Write-Up: Minimum-Stop Viewpoint Planning for Accurate Scanning

Author: Mais Jamal

## Approach

This submission is a **baseline** that I intended as a starting point to develop further, so the
emphasis is on a clean, deterministic pipeline I can then tune, rather than on a finished,
fully-tuned result.

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

1. **Candidate generation.** Lay a regular lattice of candidate stops over free space and keep
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
   and stop once the marginal gain drops below a floor, the coverage target is reached, or the
   stop budget is exhausted.
4. **Tour ordering.** Once the stops are chosen, shorten the route that visits them:
   nearest-neighbor from every possible start, then **2-opt** refinement, keeping the best
   result. Both run over **geodesic (free-space) distances** from `multi_target_shortest_paths`,
   not straight lines — the scorer reports a routed distance, so optimizing against
   straight-line distance would be optimizing the wrong objective and would mis-order any tour
   whose stops sit in different rooms.

**Determinism.** There is no randomness anywhere in the pipeline. The candidate lattice is
generated in row-major order, and every tie — in greedy selection, in component choice, in
nearest-neighbor, in 2-opt — is broken by the lowest candidate index. The same map and sensor
therefore always produce the same stops in the same order.

**What I observed while tuning.** The first runs gave a genuinely small stop count but low
coverage (~20–30%). I then swept the parameters — reducing the spacing between candidate stops
well below `r_eff` so that greedy has real choice about where to stand, changing how the
component is selected, and raising the stop budget — and found that coverage across all maps
does not exceed roughly 70% even at the most permissive settings. That ceiling is not a
shortcoming of the search; it is structural, and I explain where it comes from below.

## Design Decisions & Tradeoffs

The rubric ranks **coverage > stop count > tour length**, so coverage is where the budget should
go. The parameter set as currently committed is tuned conservatively, though: it optimizes stop
count and tour length harder than the rubric asks for, and rebalancing that is the main thing I
would change next.

**Candidate spacing (0.75 m).** Well under `r_eff ≈ 5.66 m`. Spacing candidates at `r_eff` — one
stop per sensor footprint — was my first instinct, but it makes the lattice too coarse for
greedy to have any real choice, and a candidate that happens to land in a doorway or behind a
pillar then has no nearby alternative. Denser candidates cost planning time (a scan is ~5 ms and
the whole lattice is scanned once) but buy a much better greedy trajectory. This is the
coverage-vs-runtime dial, and runtime is not scored, so I biased toward density.

**Marginal-gain floor measured in rays, not in wall cells.** This one is easy to get wrong. Every
ray terminates on exactly one wall cell, so a single stop can credit **at most `num_rays` = 720
cells**, no matter how much wall is geometrically in view. On a map with ~55k observable wall
cells that ceiling is only 1.3% of the map — so a threshold like "a stop must add at least 2% of
the total wall" would reject *every possible stop* and return an empty plan. The floor is
therefore expressed as a fraction of `num_rays` (currently 0.35, i.e. 252 cells). This is the
direct stop-count-vs-coverage dial: raising it yields fewer, denser stops; lowering it keeps
buying coverage at a progressively worse rate per stop.

**Restricting the plan to one connected component.** This is the decision with the largest cost,
and it is a deliberate trade of coverage for *honesty*. These maps are 16-bit PGMs; PIL opens
them as mode `"I"` and `.convert("L")` in `map_io` clips rather than scales, so the "unknown"
gray saturates to 255 and loads as **FREE**. The consequence is that the unmapped area outside
the building reads as drivable, and free space fragments into many disconnected components
(9 on map 1, 72 on map 2) separated by exterior walls. If stops are allowed to spread across
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

**Where the coverage ceiling actually comes from.** The scorer's denominator is *every* observable
wall cell on the map — including the outward faces of exterior walls, which are "observable" only
because the unmapped area beyond them was mis-read as free space. Those faces are not reachable
from inside the building at all. Combined with the single-component restriction, a large share of
the denominator is unreachable by construction, which is why no parameter setting pushes coverage
past ~70%. The ~23–25% currently reported is therefore the product of two separate things: this
structural ceiling, and a marginal-gain floor and stop budget that are set too conservatively for
a coverage-first rubric. Only the second is something I can move, and it is where I would start.

## Conclussions

### Performance

### What I'd Do With More Time

### Known Limitations / Where I Expect This to Break


## Reproducibility

<!-- If your solution relies on any randomness (random restarts, stochastic
sampling, seeds, etc.), state the seed(s) used and confirm re-running
produces identical results. -->

## Modified/Added Files or Packages (optional)

<!-- If you touched anything beyond the expected solution file (e.g. extra
dependencies, new modules, Nav2 config, helper scripts), list them here with
a short reason why. -->
