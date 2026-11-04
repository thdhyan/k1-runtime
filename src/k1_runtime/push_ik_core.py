"""Shared-actor controller for Mjlab-PushIK-Booster-K1, without ROS or mjlab: mocap poses -> per-robot commands.

Must match tasks_push_ik.py (checked by test/check_push_ik_parity.py inside the training env).
Pose = (x, y, yaw) of each robot TRUNK and of the BOX CENTRE, in one world frame, at 50 Hz.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

from k1_runtime import wbc  # noqa: E402

HISTORY, ACTION_DIM = 5, 10
CMD_SCALE = np.array((0.5, 0.3, 0.8))
HEIGHT_RANGE = 0.12
HAND_CENTER, HAND_SCALE = (0.27, 0.2, -0.1), (0.13, 0.12, 0.2)
CMD_ALPHA, HAND_ALPHA = 0.2, 0.2
SUCCESS_R, SUCCESS_YAW = 0.2, math.radians(10.0)
SIZE_RANGE = ((0.2, 0.35), (0.4, 0.75), (0.15, 0.3))


def box_errors(box, goal) -> tuple[float, float]:
  """Position error (m) and yaw error (rad, box is 180-deg symmetric)."""
  e = box[2] - goal[2]
  return math.hypot(box[0] - goal[0], box[1] - goal[1]), abs(math.atan2(math.sin(2 * e), math.cos(2 * e))) / 2


def _rel(me, p):
  dx, dy, c, s = p[0] - me[0], p[1] - me[1], math.cos(me[2]), math.sin(me[2])
  return [c * dx + s * dy, -s * dx + c * dy]


def _axis(yaw):
  return [math.sin(2 * yaw), math.cos(2 * yaw)]


class PushIKController:
  def __init__(self, ckpt: str, half_size, device="cpu"):
    self.policy = wbc.PolicyModule(ckpt, device)
    self.half, self.device = list(map(float, half_size)), device
    c, s = torch.tensor(HAND_CENTER), torch.tensor(HAND_SCALE)
    flip = torch.tensor((1.0, -1.0, 1.0))
    self.center, self.scale = torch.cat([c, c * flip]), torch.cat([s, s * flip])
    self.ik = wbc.ArmIK(2, rate=3.0, dt=0.02)
    self.reset()

  def reset(self):
    self.hist = None
    self.last = np.zeros((2, ACTION_DIM))
    self.cmd = np.zeros((2, 3))
    self.goal = self.center.repeat(2, 1)
    self.ik.reset()

  def obs(self, a, b, box, goal) -> torch.Tensor:
    """(2, 125): per agent box 4, goal 4, mate 4, size 3, last action 10, each oldest to newest over HISTORY steps."""
    now = []
    for k, (me, mate) in enumerate(((a, b), (b, a))):
      dy = mate[2] - me[2]
      now.append([_rel(me, box) + _axis(box[2] - me[2]), _rel(me, goal) + _axis(goal[2] - me[2]),
                  _rel(me, mate) + [math.sin(dy), math.cos(dy)], self.half, list(self.last[k])])
    if self.hist is None:
      self.hist = [now] * HISTORY
    self.hist = self.hist[1:] + [now]
    # (agent, term) -> history joined per term, the layout of a flattened mjlab history group
    out = [[v for t in range(5) for h in self.hist for v in h[k][t]] for k in range(2)]
    return torch.tensor(out, dtype=torch.float32)

  def act(self, obs: torch.Tensor) -> np.ndarray:
    return self.policy(obs).numpy()

  def decode(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """raw (2, 10) -> cmd (2, 3), trunk height offset (2,), arm targets (2, 8). Same filters as the training actions."""
    self.last = raw.astype(float)
    self.cmd += CMD_ALPHA * (np.clip(raw[:, :3], -1, 1) * CMD_SCALE - self.cmd)
    dh = HEIGHT_RANGE * np.clip(raw[:, 3], -1, 0)
    goal = self.center + self.scale * torch.as_tensor(raw[:, 4:], dtype=torch.float32).clamp(-1, 1)
    self.goal = self.goal + HAND_ALPHA * (goal - self.goal)
    return self.cmd.copy(), dh, self.ik.step(self.goal).numpy().copy()

  def step(self, a, b, box, goal):
    return self.decode(self.act(self.obs(a, b, box, goal)))
