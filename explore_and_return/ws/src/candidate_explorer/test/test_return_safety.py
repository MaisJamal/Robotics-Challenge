"""Offline regressions; run with python3 -m unittest discover -s <this directory>."""
import ast
from concurrent.futures import Future
from types import SimpleNamespace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1] / 'candidate_explorer'
sys.path.insert(0, str(PACKAGE.parent))
from candidate_explorer.ra_ufe import Planner, PlannerConfig, dijkstra

# Exercise the actual node methods without requiring a ROS installation.
# ROS integration (TF, transport, executor ordering) still needs simulation.
tree = ast.parse((PACKAGE / 'explorer_node.py').read_text())
node_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExplorerNode')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node_class], type_ignores=[])
namespace = dict(Node=object, math=math, Twist=lambda: 'STOP',
                 NavigateToPose=SimpleNamespace(Goal=SimpleNamespace),
                 Trigger=SimpleNamespace(Request=SimpleNamespace))
exec(compile(ast.fix_missing_locations(module), str(PACKAGE / 'explorer_node.py'), 'exec'), namespace)
ExplorerNode = namespace['ExplorerNode']


class ReturnSafetyTests(unittest.TestCase):
    def test_no_diagonal_corner_cutting(self):
        mask = np.eye(2, dtype=bool)
        self.assertTrue(math.isinf(dijkstra(mask, np.ones((2, 2)), [(0, 0)], 1)[1, 1]))

    def test_symmetric_cost_and_blocked_source(self):
        mask = np.ones((1, 2), bool)
        weights = np.array([[1., 3.]])
        self.assertEqual(dijkstra(mask, weights, [(0, 0)], 1)[0, 1],
                         dijkstra(mask, weights, [(0, 1)], 1)[0, 0])
        mask[0, 0] = False
        self.assertTrue(np.isinf(dijkstra(mask, weights, [(0, 0)], 1)).all())

    def test_unknown_or_wall_cannot_verify_return(self):
        planner = Planner(PlannerConfig(target_cell_m=.1))
        for barrier in (-1, 100):
            data = np.zeros((40, 60), dtype=int)
            data[:, 29:31] = barrier
            result = planner.plan(planner.build_view(data, .1, 0, 0), (1., 2.), 0., (5., 2.))
            self.assertFalse(result.home_reachable)
            self.assertTrue(math.isinf(result.robot_home_cost))
            self.assertEqual(result.candidates, [])

    def test_known_return_and_bootstrap_still_work(self):
        planner = Planner(PlannerConfig(target_cell_m=.1))
        data = np.full((60, 60), -1, dtype=int)
        data[27:33, 27:33] = 0
        result = planner.plan(planner.build_view(data, .1, 0, 0), (3., 3.), 0., (3., 3.))
        self.assertTrue(result.home_reachable)
        self.assertTrue(result.candidates)
        self.assertTrue(all(c.source == 'probe' and math.isfinite(c.home_cost) for c in result.candidates))

    def test_unknown_start_cell_can_probe_only_when_confirmed_home(self):
        planner = Planner(PlannerConfig(target_cell_m=.1))
        data = np.full((60, 60), -1, dtype=int)
        data[28:30, 28:30] = 0
        view = planner.build_view(data, .1, 0, 0)
        uncertain = planner.plan(view, (3., 3.), 0., (3., 3.))
        self.assertFalse(uncertain.home_reachable)
        confirmed = planner.plan(view, (3., 3.), 0., (3., 3.), at_home=True)
        self.assertTrue(confirmed.home_reachable)
        self.assertEqual(confirmed.robot_home_cost, 0.)
        self.assertTrue(confirmed.candidates)
        self.assertTrue(all(c.source == 'probe' for c in confirmed.candidates))

    def test_fine_return_survives_conservative_downsampling(self):
        planner = Planner(PlannerConfig(target_cell_m=.1))
        data = np.full((60, 100), -1, dtype=int)
        # The known corridor straddles coarse block boundaries: no coarse
        # cell is wholly free, although a continuous fine route exists.
        data[29:33, 5:95] = 0
        view = planner.build_view(data, .025, 0, 0)
        result = planner.plan(view, (2., .775), 0., (.25, .775))
        self.assertTrue(result.home_reachable)
        self.assertTrue(math.isfinite(result.robot_home_cost))
        # An actual unknown break must still invalidate that route.
        data[:, 48:52] = -1
        result = planner.plan(planner.build_view(data, .025, 0, 0),
                              (2., .775), 0., (.25, .775))
        self.assertFalse(result.home_reachable)
        self.assertTrue(math.isinf(result.robot_home_cost))

    def test_elapsed_pressure_increases_and_can_be_disabled(self):
        planner = Planner(PlannerConfig())
        reserve = planner.reserve_for_return(2.)
        self.assertAlmostEqual(planner.time_pressure(2., 0., 5400.), reserve / 5400.)
        self.assertAlmostEqual(planner.time_pressure(2., 1800., 5400.), reserve / 3600. + .5)
        self.assertEqual(planner.time_pressure(2., 5400., 5400.), 1.)
        planner.cfg.time_pressure_gain = 0.
        self.assertAlmostEqual(planner.time_pressure(2., 1800., 5400.), reserve / 3600.)

    def test_return_accepts_distance_that_rounds_to_point_fifteen(self):
        node = object.__new__(ExplorerNode)
        node._goal_in_progress = False
        node.planner = SimpleNamespace(now_s=0.)
        node.elapsed_s = lambda: 100.
        node.remaining_s = lambda: 5000.
        node.finish_reserve_s = 60.
        node.distance_to_home_m = lambda: .1508
        node.home_tolerance_m = .20
        node.get_logger = Mock()
        node.state = 'RETURNING'
        node._return_tick()
        self.assertEqual(node.state, 'FINISHING')

    def test_failed_grade_still_acknowledges_finalization(self):
        node = object.__new__(ExplorerNode)
        node.state = 'FINISHING'
        node.finish_client = Mock()
        node.get_logger = Mock()
        response = Future()
        node.finish_client.call_async.return_value = response
        node.call_finish_exploration()
        response.set_result(SimpleNamespace(success=False, message='success=False report_written_to=/tmp/report.yaml'))
        self.assertEqual(node.state, 'DONE')

    def test_deadline_cancels_active_goal_and_recovery(self):
        node = object.__new__(ExplorerNode)
        node.state = 'RETURNING'
        node.remaining_s = lambda: 10.
        node.finish_reserve_s = 60.
        node.stop_unstick = Mock()
        node.cancel_goal = Mock()
        node.unsticking = lambda: False
        node._nav_outstanding = {1}
        node._return_tick = Mock()
        node._tick()
        self.assertEqual(node.state, 'FINISHING')
        node.stop_unstick.assert_called_once()
        node.cancel_goal.assert_called_once()
        node._return_tick.assert_not_called()

    def test_recovery_expiry_publishes_stop_once(self):
        node = object.__new__(ExplorerNode)
        node._unstick_active = True
        node._unstick_driving = True
        node._unstick_until_s = 5.
        node._nav_outstanding = set()
        node.state = 'RETURNING'
        node.elapsed_s = lambda: 6.
        node.remaining_s = lambda: 100.
        node.finish_reserve_s = 60.
        node.cmd_pub = Mock()
        node._unstick_tick()
        node._unstick_tick()
        node.cmd_pub.publish.assert_called_once_with('STOP')
        self.assertFalse(node.unsticking())

    def test_recovery_waits_for_terminal_navigation_result(self):
        node = object.__new__(ExplorerNode)
        node._unstick_active = True
        node._unstick_driving = False
        node._unstick_wait_until_s = 5.
        node._nav_outstanding = {1}
        node.state = 'RETURNING'
        node.elapsed_s = lambda: 2.
        node.remaining_s = lambda: 100.
        node.finish_reserve_s = 60.
        node.cmd_pub = Mock()
        node._unstick_tick()
        node.cmd_pub.publish.assert_not_called()

    def test_late_acceptance_is_cancelled_and_waited_for(self):
        node = object.__new__(ExplorerNode)
        node.nav_client = Mock()
        accepted, terminal = Future(), Future()
        node.nav_client.send_goal_async.return_value = accepted
        node._goal_seq = 0
        node._nav_outstanding = set()
        node._goal_handle = None
        done = Mock()
        node.send_nav_goal('pose', done)
        node.cancel_goal()
        handle = Mock(accepted=True)
        handle.get_result_async.return_value = terminal
        accepted.set_result(handle)
        handle.cancel_goal_async.assert_called_once()
        self.assertTrue(node._nav_outstanding)
        terminal.set_result(SimpleNamespace(status=5))
        self.assertFalse(node._nav_outstanding)
        done.assert_not_called()

    def test_finish_waits_for_acknowledgment_and_retries_failure(self):
        node = object.__new__(ExplorerNode)
        node.state = 'FINISHING'
        node._finish_sent = True
        node.finish_client = Mock()
        node.get_logger = Mock()
        response = Future()
        node.finish_client.call_async.return_value = response
        node.call_finish_exploration()
        self.assertEqual(node.state, 'FINISHING')
        response.set_result(SimpleNamespace(success=False, message='retry'))
        self.assertFalse(node._finish_sent)
        response = Future()
        node.finish_client.call_async.return_value = response
        node.call_finish_exploration()
        response.set_result(SimpleNamespace(success=True, message='finished'))
        self.assertEqual(node.state, 'DONE')


if __name__ == '__main__':
    unittest.main()
