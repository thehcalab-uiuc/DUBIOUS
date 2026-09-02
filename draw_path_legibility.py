# THIS FILE WAS ENTIRELY WRITTEN BY COPILOT
"""Interactive freehand path drawing and legibility scoring.

This script reuses `new_sim/path_evaluator.py` for the score computation.
Draw a path with the mouse, press Enter, and it will report the legibility
score for the drawn trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from costfunctions.dragancost import DraganCostFunction
from environment import env
from path_evaluator import PathEvaluator


DEFAULT_ENV_FILE = SCRIPT_DIR / "experiment.json"


def load_environment(env_json_path: Path | None) -> env:
    path = env_json_path or DEFAULT_ENV_FILE
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return env.from_json_dict(data["env"] if "env" in data else data)


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    if len(points) < 2 or step <= 0:
        return points

    deltas = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]

    if total_length <= 0:
        return points[:1]

    sample_positions = np.arange(0.0, total_length, step)
    if sample_positions.size == 0 or sample_positions[-1] != total_length:
        sample_positions = np.append(sample_positions, total_length)

    xs = np.interp(sample_positions, cumulative_lengths, points[:, 0])
    ys = np.interp(sample_positions, cumulative_lengths, points[:, 1])
    return np.column_stack((xs, ys))


class InteractivePathDrawer:
    def __init__(self, environment: env, min_point_distance: float = 0.04):
        self.env = environment
        self.min_point_distance = min_point_distance
        self.points: list[list[float]] = []
        self.is_drawing = False
        self.finished = False

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.env.display(self.ax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Draw a path with the left mouse button. Enter to score. r to reset. q to quit.")

        (self.line,) = self.ax.plot([], [], color="tab:blue", linewidth=2, marker="o", markersize=3)
        self.status_text = self.ax.text(
            0.01,
            0.01,
            "Click and drag to draw.",
            transform=self.ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_plot(self) -> None:
        if self.points:
            pts = np.asarray(self.points)
            self.line.set_data(pts[:, 0], pts[:, 1])
            self.status_text.set_text(f"Points: {len(self.points)}")
        else:
            self.line.set_data([], [])
            self.status_text.set_text("Click and drag to draw.")
        self.fig.canvas.draw_idle()

    def _append_point(self, x: float, y: float, force: bool = False) -> None:
        if not np.isfinite(x) or not np.isfinite(y):
            return

        new_point = np.array([x, y], dtype=float)
        if not self.points or force:
            self.points.append(new_point.tolist())
            self._update_plot()
            return

        last_point = np.asarray(self.points[-1], dtype=float)
        if np.linalg.norm(new_point - last_point) >= self.min_point_distance:
            self.points.append(new_point.tolist())
            self._update_plot()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1:
            return
        self.is_drawing = True
        self._append_point(event.xdata, event.ydata, force=True)

    def _on_motion(self, event) -> None:
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self._append_point(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        if event.button == 1:
            self.is_drawing = False

    def _reset(self) -> None:
        self.points = []
        self.is_drawing = False
        self.finished = False
        self._update_plot()

    def _finish(self) -> None:
        self.finished = True
        plt.close(self.fig)

    def _on_key(self, event) -> None:
        if event.key in {"enter", "return"}:
            self._finish()
        elif event.key == "r":
            self._reset()
        elif event.key == "q":
            self.points = []
            self._finish()

    def show(self) -> np.ndarray:
        plt.show()
        return np.asarray(self.points, dtype=float)


def choose_goal_index(environment: env, path: np.ndarray, requested_goal_idx: int | None) -> int:
    if requested_goal_idx is not None:
        return int(requested_goal_idx)
    return int(np.argmin(np.linalg.norm(environment.goals - path[-1], axis=1)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a path in an environment and compute its legibility score.")
    parser.add_argument("--env-json", type=Path, default=DEFAULT_ENV_FILE, help="Path to an environment JSON file.")
    parser.add_argument("--goal-idx", type=int, default=None, help="Goal index to score against. Defaults to nearest goal to the drawn path end.")
    parser.add_argument("--min-point-distance", type=float, default=0.04, help="Minimum distance between stored mouse points.")
    parser.add_argument("--resample-step", type=float, default=0.05, help="Resample spacing for the final path before scoring.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional JSON file to save the drawn path and score.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    environment = load_environment(args.env_json)

    drawer = InteractivePathDrawer(environment, min_point_distance=args.min_point_distance)
    raw_path = drawer.show()

    if len(raw_path) < 2:
        print("No path was drawn.")
        return 1

    final_path = resample_polyline(raw_path, args.resample_step)
    goal_idx = choose_goal_index(environment, final_path, args.goal_idx)
    evaluation_path = final_path.copy()
    evaluation_path[-1] = environment.goals[goal_idx]

    evaluator = PathEvaluator(
        env=environment,
        path=evaluation_path,
        pathinfo={"source": "interactive_draw"},
        evaluator_costfn=DraganCostFunction(environment),
    )
    legibility_scores = evaluator.calculate_legibility_score(goal_idx=goal_idx)

    print(f"Target goal index: {goal_idx}")
    print(f"Legibility scores [total, observers...]: {np.array2string(legibility_scores, precision=6)}")

    if args.save_path is not None:
        payload = {
            "env": environment.to_json_dict(),
            "path": final_path.tolist(),
            "evaluation_path": evaluation_path.tolist(),
            "goal_idx": int(goal_idx),
            "legibility_scores": np.asarray(legibility_scores).tolist(),
        }
        with open(args.save_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
        print(f"Saved path and score to {args.save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())