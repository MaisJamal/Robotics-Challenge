# Challenge: Autonomous Explore-and-Return

## The problem

Survey a space you've never seen, as fast as possible, then get back to
where you started. You control a simulated robot with a 2D laser scanner
(no camera, no known map) dropped at a random point inside a real floor
plan. Build a map, explore it, decide when you've seen enough, and return
home — autonomously, with no human in the loop.

## What's provided vs. what you build

You are **not** implementing SLAM — that part is provided, running, and not the point of this exercise:

| Provided (don't touch) | You write |
|---|---|
| The simulator (laser + odometry + collision physics) | One ROS 2 node: `candidate_explorer` |
| Live SLAM (`slam_toolbox`) — publishes `/map` as you drive | Exploration strategy: where to go next |
| Navigation (Nav2) — given a goal pose, it drives there and avoids obstacles | Coverage/termination logic: when are you done? |
| | Return-home logic: get back and say so |

Two pieces of Nav2 config are the exception and *are* candidate-editable —
the behavior tree (`candidate_explorer/config/navigate_to_pose.xml`) and the
costmap params (`candidate_explorer/config/costmap_params.yaml`) — because
they shape exploration throughput rather than the core driving/avoidance
behavior. If you'd rather build your own navigation instead of using Nav2,
that's fine too, but expect more work.

Your node's contract is exactly three things:
1. Subscribe `/map` (`nav_msgs/OccupancyGrid`), published live by `slam_toolbox`.
2. Send goals via the `/navigate_to_pose` action (`nav2_msgs/action/NavigateToPose`).
3. Call the `/finish_exploration` service (`std_srvs/srv/Trigger`) when you
   believe the survey is complete and you're home.

Everything else in `candidate_explorer/explorer_node.py` is scaffolding you
can keep, restructure, or delete — only that contract matters.

## Getting started

The whole setup is containerized. From the root of explore_and_return challenge:

```bash
docker compose -f docker/docker-compose.yml up -d
docker exec -it explore_and_return_challenge bash
```

Inside the container, build and source the workspace:

```bash
colcon build --symlink-install && source install/setup.bash
```

If you want to open RViz or RQt from inside the container, first run this
on the **host** so the container can reach your X server:

```bash
xhost +local:docker
```

## Running the challenge

Two terminals:

```bash
# Terminal 1: the provided infrastructure
ros2 launch challenge_sim challenge.launch.py \
  map_yaml:=../maps/1/room.yaml seed:=1 rviz:=true

# Terminal 2, after a few seconds for bringup:
ros2 run candidate_explorer explorer_node
```

Or use the convenience script (run from `/challenge`):

```bash
./eval_runner.sh maps/1/room.yaml 1
```

Everything runs on **sim time**, not the wall clock, so you can iterate fast.

### Arguments

All of these are `challenge.launch.py` arguments (`name:=value`); `eval_runner.sh`
takes the first four positionally (`eval_runner.sh <map_yaml> [seed] [time_limit_s]
[time_scale]`) and forwards them, plus an `RVIZ=true` environment variable for the last one.

| Argument | Default | Meaning |
|---|---|---|
| `map_yaml` | *required* | Path to the ground-truth `room.yaml`. Never exposed to your node. |
| `seed` | `0` | RNG seed for the spawn point. `0` = random each run. Any other integer reproduces the same spawn — use a fixed seed while developing, expect evaluation to vary it. |
| `time_limit_s` | `5400.0` | Sim-time budget in seconds (1.5h) — see "Speed and time limits." |
| `time_scale` | `4.0` | How much faster than real time the whole stack runs — see "Speed and time limits." |
| `results_path` | `results/<timestamp>/report.yaml` | Where the report (and `map.pgm`/`map.yaml`) are written. |
| `rviz` | `false` | Set `true` to also launch RViz with the saved view. |
| `bt_xml` | `candidate_explorer/config/navigate_to_pose.xml` | Which behavior tree `bt_navigator` loads. |
| `costmap_params` | `candidate_explorer/config/costmap_params.yaml` | `local_costmap`/`global_costmap` params. |

### Watching performance live

```bash
ros2 launch challenge_sim challenge.launch.py map_yaml:=... seed:=1 rviz:=true
# or: RVIZ=true ./eval_runner.sh maps/1/room.yaml
```

The saved view shows the live map, laser scan, both costmaps, TF, and the
global/local plan, plus RViz's "2D Goal Pose" tool so you can click to send
a manual goal without writing code.

### Manual operation

Useful for sanity-checking the sim, or watching coverage happen by hand
before writing a planner:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

This publishes directly to `/cmd_vel` and bypasses your explorer node
entirely — not part of the graded task.

### Speed and time limits

- **Robot speed:** capped at 0.3 m/s linear, 1.2 rad/s angular.
- **Time limit (`time_limit_s`, default 5400s = 1.5h):** a budget of
  **simulated**, not wall-clock, time. If `/finish_exploration` hasn't been
  called by then, the session force-finishes with `timed_out: true` and
  `success: false`.
- **`time_scale` (default 4.0):** `challenge_sim` is the sole publisher of
  `/clock`, and every other node (slam_toolbox, Nav2, RViz) runs with
  `use_sim_time: true`, so `time_scale` speeds up the entire stack together,
  not just the robot. A 1.5h sim-time session at `time_scale:=4.0` takes
  about 22 real minutes; `time_scale:=1.0` runs at true real-time speed.
  Please don't go beyond **10x**. If the costmaps or TF visibly lag behind
  the robot in RViz, bring `time_scale` down.

## Results

Every session writes one YAML report to `results_path`
(`results/<timestamp>/report.yaml` by default). The directory is created
automatically. Repeat runs never clobber each other since each gets its own
timestamped folder by default.

Alongside `report.yaml`, the same directory gets `map.pgm`/`map.yaml` — your
own SLAM-built map as it looked when the session finished (not the ground
truth) — useful for a quick visual sanity check without RViz open.

The report only appears once a session finishes (via `/finish_exploration`
or the time limit) — an empty `results/` just means no session has finished yet.

### Home is the odom frame's origin

The robot's `odom` frame originates at its spawn point, exactly like a real
robot's odometry frame originates wherever it was powered on. So "home" is
always `(0, 0, 0)` in the odom frame — no special topic reveals it.

To send a return-home goal, transform that point into the `map` frame using
the *current* `map` → `odom` transform (`get_home_pose_in_map_frame()` in
the stub does this). Don't cache the map-frame pose once at startup:
`slam_toolbox` corrects `map` → `odom` as it refines its pose graph, so
where home sits in the map frame can drift over a run even though the robot
never moved in the odom frame. Getting this wrong is a realistic,
easy-to-miss bug, not a trick question.

### Scoring

When you call `/finish_exploration` (or the time limit expires), a report is
written with:

- **`coverage_fraction`** — fraction of reachable free space your laser
  actually observed. The primary metric.
- **`distance_to_home_m`** — how far from true home you ended up.
- **`elapsed_time_s`** — simulated elapsed time, graded against
  `time_limit_s`. Independent of `time_scale`.
- **`elapsed_wall_time_s`** — real seconds the session took, for reference only.
- **`collision_count`** — times a commanded move was blocked by a wall.
- **`success`** — `true` only if you finished before the time limit, reached
  ≥80% coverage, and ended within 0.3 m of home.

There's a real tradeoff between exploring thoroughly and returning quickly.
Say what you optimized for in your write-up.

### What "good" looks like

A baseline: detect frontiers (boundaries between known-free and unknown
space in `/map`), pick the nearest one, navigate to it, repeat until no
frontiers remain, then go home. That's a legitimate, gradeable submission —
essentially what `explore_lite`, a well-known ROS package, does.

Stronger submissions might:
- Cluster frontiers and cost them by more than just distance (size,
  direction, expected information gain) rather than pure nearest-first.
- Recognize when you're revisiting already-covered ground and correct course.
- Distinguish "no more frontiers" from "good enough" — you don't need 100%
  coverage if the marginal cost of the last few percent is high.
- Handle a stuck/oscillating episode gracefully instead of looping forever.

## Write-Up

In [write-ups](../write-ups/) document your approach and reasoning behind it, what you'd do with more time,
and where you expect it to break. you can also use the template located in the directory.
