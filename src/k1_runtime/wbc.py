"""Decoupled whole-body controller (HRL WP0 contract + runtime), torch only: no mjlab/Isaac import.

Layers (PLAN_HRL_CONTROLLER.md): task policy -> commands -> {lower body, head, arms} -> joint PD targets.
Each body module owns only its joints and is frozen; modules talk only through the command dataclasses:

  LocoCmd(vx, vy, wz[, dh]) -> LocoModule (mjlab walk base, 225 obs; 230 with the height channel) -> 12 leg targets
  HeadCmd(yaw, pitch)  -> head joints directly, or HeadModule (ball-track policy) -> 2 head targets
  ArmCmd(q[8])         -> arm joints (hold pose, carry/push arm policy, IK, teleop) -> 8 arm targets

Joint order everywhere = MJCF model order ``JOINTS`` (head 2 | L arm 4 | R arm 4 | L leg 6 | R leg 6), the order
mjlab, the exported USD and the deploy stack use. Constants repeat k1_constants.py so this file runs on the robot.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

JOINTS = (
  "AAHead_yaw", "Head_pitch",
  "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
  "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
  "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
  "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
HEAD, ARMS, LEGS = slice(0, 2), slice(2, 10), slice(10, 22)
#: k1_constants.HOME (knees bent, arms down)
HOME = (0.0, 0.0, 0.0, -1.3, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0) + (-0.3, 0.0, 0.0, 0.6, -0.3, 0.0) * 2
#: MJCF joint ranges (rad)
LIMITS = (
  (-1.0, 1.0), (-0.349, 0.855),
  (-3.316, 1.22), (-1.74, 1.57), (-2.27, 2.27), (-2.44, 0.0),
  (-3.316, 1.22), (-1.57, 1.74), (-2.27, 2.27), (0.0, 2.44),
  (-3.0, 2.21), (-0.4, 1.57), (-1.0, 1.0), (0.0, 2.23), (-0.87, 0.345), (-0.345, 0.345),
  (-3.0, 2.21), (-1.57, 0.4), (-1.0, 1.0), (0.0, 2.23), (-0.87, 0.345), (-0.345, 0.345),
)
#: trunk height over the feet when standing (tasks_skills.H_STAND) and the squat floor (SQUAT_END[0])
H_STAND, H_MIN = 0.52, 0.35
_W2 = (4.0 * 2.0 * math.pi) ** 2
#: walk-base action scale per leg joint = 0.25 * effort / kp (k1_constants.K1_ACTION_SCALE)
LEG_SCALE = tuple(0.25 * e / (a * _W2) for e, a in (
  (68.0, 0.0478125), (76.0, 0.0339552), (38.3, 0.0282528), (112.0, 0.095625),
  (38.3, 0.0282528 * 1.4), (38.3, 0.0282528 * 0.4))) * 2

#: left/right mirror: index permutation + sign (roll/yaw axes flip, pitch axes keep)
_SWAP = list(range(2)) + list(range(6, 10)) + list(range(2, 6)) + list(range(16, 22)) + list(range(10, 16))
_SIGN = (-1, 1) + (1, -1, 1, -1) * 2 + (1, -1, -1, 1, 1, -1) * 2


def mirror_joints(q: torch.Tensor) -> torch.Tensor:
  """(..., 22) joint vector -> its left/right mirror image."""
  return q[..., _SWAP] * torch.tensor(_SIGN, dtype=q.dtype, device=q.device)


class SkillToken(enum.IntEnum):
  """Discrete skill the task layer selects (hybrid action head, WP7)."""
  IDLE = 0
  WALK = 1
  KICK = 2
  PUSH = 3
  CARRY = 4


@dataclass
class LocoCmd:
  """Base velocity command, body frame, + trunk height offset dh from H_STAND (0 = stand; height bases only).
  Clamped to the envelope the walk base was trained on."""
  vx: torch.Tensor
  vy: torch.Tensor
  wz: torch.Tensor
  dh: torch.Tensor | None = None
  RANGE = ((-1.0, 1.4), (-0.5, 0.5), (-1.2, 1.2))

  def tensor(self) -> torch.Tensor:
    v = torch.stack([self.vx, self.vy, self.wz], dim=-1)
    lo, hi = torch.tensor(self.RANGE, dtype=v.dtype, device=v.device).T
    return torch.maximum(torch.minimum(v, hi), lo)

  def height(self) -> torch.Tensor:
    """(n, 1) absolute trunk-height command, the squat task's H*."""
    dh = torch.zeros_like(self.vx) if self.dh is None else self.dh
    return (H_STAND + dh).clamp(H_MIN, H_STAND)[..., None]

  @classmethod
  def from_action(cls, a: torch.Tensor, scale=(1.0, 0.5, 1.0)) -> "LocoCmd":
    """Task-layer action in [-1, 1] (FrozenBaseActionCfg.cmd_scale) -> command."""
    v = a[..., :3].clamp(-1.0, 1.0) * torch.tensor(scale, dtype=a.dtype, device=a.device)
    return cls(v[..., 0], v[..., 1], v[..., 2])


@dataclass
class HeadCmd:
  """Head joint targets (yaw, pitch), rad; clamped to the joint range."""
  q: torch.Tensor  # (n, 2)

  def tensor(self) -> torch.Tensor:
    lim = torch.tensor(LIMITS[HEAD], dtype=self.q.dtype, device=self.q.device)
    return torch.maximum(torch.minimum(self.q, lim[:, 1]), lim[:, 0])


@dataclass
class ArmCmd:
  """Arm joint targets (L 4 | R 4), rad; clamped to the joint range."""
  q: torch.Tensor  # (n, 8)

  def tensor(self) -> torch.Tensor:
    lim = torch.tensor(LIMITS[ARMS], dtype=self.q.dtype, device=self.q.device)
    return torch.maximum(torch.minimum(self.q, lim[:, 1]), lim[:, 0])


@dataclass
class Proprio:
  """What the body modules read each 50 Hz step (robot IMU + joint encoders), model joint order."""
  ang_vel: torch.Tensor   # (n, 3) trunk gyro, body frame
  gravity: torch.Tensor   # (n, 3) gravity unit vector, body frame
  q: torch.Tensor         # (n, 22)
  dq: torch.Tensor        # (n, 22)


def load_frozen_mlp(ckpt: str, device) -> tuple[nn.Sequential, torch.Tensor, torch.Tensor]:
  """rsl-rl checkpoint -> (mlp, obs mean, obs std+eps). ELU MLP, normaliser eps 1e-2."""
  sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
  layers, i = [], 0
  while f"mlp.{i}.weight" in sd:
    w = sd[f"mlp.{i}.weight"]
    lin = nn.Linear(w.shape[1], w.shape[0])
    lin.weight.data, lin.bias.data = w, sd[f"mlp.{i}.bias"]
    layers.append(lin)
    if f"mlp.{i + 2}.weight" in sd:
      layers.append(nn.ELU())
    i += 2
  mlp = nn.Sequential(*layers).to(device).eval()
  for p in mlp.parameters():
    p.requires_grad_(False)
  return mlp, sd["obs_normalizer._mean"].to(device), (sd["obs_normalizer._std"] + 1e-2).to(device)


class PolicyModule:
  """Frozen normalised MLP: obs -> action."""

  def __init__(self, ckpt: str, device="cpu"):
    self.mlp, self.mean, self.std = load_frozen_mlp(ckpt, device)

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp((obs - self.mean) / self.std)


class LocoModule:
  """Walk base: LocoCmd + proprio -> 12 leg targets. Obs = 5-step term-major history of
  ang 3 | grav 3 | leg q-home 12 | leg dq 12 | last action 12 | cmd 3 [| height cmd 1] (mjlab Velocity actor obs;
  the height term exists for Mjlab-Velocity-Height-* bases, detected from the checkpoint's obs size)."""
  HISTORY = 5

  def __init__(self, ckpt: str, num_envs: int = 1, device="cpu"):
    self.policy = PolicyModule(ckpt, device)
    self.TERMS = (3, 3, 12, 12, 12, 3) + ((1,) if self.policy.mean.shape[-1] == 46 * self.HISTORY else ())
    self.home = torch.tensor(HOME[LEGS], device=device)
    self.scale = torch.tensor(LEG_SCALE, device=device)
    self.last = torch.zeros(num_envs, 12, device=device)
    self.hist = torch.zeros(num_envs, self.HISTORY, sum(self.TERMS), device=device)
    self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

  def reset(self, ids=slice(None)) -> None:
    self.last[ids] = 0.0
    self.fresh[ids] = True

  def obs(self, p: Proprio, cmd: LocoCmd) -> torch.Tensor:
    terms = [p.ang_vel, p.gravity, p.q[:, LEGS] - self.home, p.dq[:, LEGS], self.last, cmd.tensor()]
    frame = torch.cat(terms + [cmd.height()] * (len(self.TERMS) == 7), -1)
    self.hist = torch.where(
      self.fresh[:, None, None], frame[:, None].expand_as(self.hist), torch.cat([self.hist[:, 1:], frame[:, None]], 1))
    self.fresh[:] = False
    return torch.cat([c.flatten(1) for c in self.hist.split(self.TERMS, dim=-1)], dim=-1)

  def __call__(self, p: Proprio, cmd: LocoCmd) -> torch.Tensor:
    a = self.policy(self.obs(p, cmd))
    self.last = a
    return self.home + self.scale * a


class GetUpModule:
  """Get-up (Mjlab-GetUp): proprio -> 22 targets. Obs = 5-step term-major history of
  ang 3 | grav 3 | q-home 22 | dq 22 | last action 22; target = home + scale * a (head 0.2, arms 1.0, legs LEG_SCALE)."""
  HISTORY, TERMS = 5, (3, 3, 22, 22, 22)
  SCALE = (0.2,) * 2 + (1.0,) * 8 + LEG_SCALE

  def __init__(self, ckpt: str, num_envs: int = 1, device="cpu"):
    self.policy = PolicyModule(ckpt, device)
    self.home = torch.tensor(HOME, device=device)
    self.scale = torch.tensor(self.SCALE, device=device)
    self.last = torch.zeros(num_envs, 22, device=device)
    self.hist = torch.zeros(num_envs, self.HISTORY, sum(self.TERMS), device=device)
    self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

  def reset(self, ids=slice(None)) -> None:
    self.last[ids] = 0.0
    self.fresh[ids] = True

  def obs(self, p: Proprio) -> torch.Tensor:
    frame = torch.cat([p.ang_vel, p.gravity, p.q - self.home, p.dq, self.last], -1)
    self.hist = torch.where(
      self.fresh[:, None, None], frame[:, None].expand_as(self.hist), torch.cat([self.hist[:, 1:], frame[:, None]], 1))
    self.fresh[:] = False
    return torch.cat([c.flatten(1) for c in self.hist.split(self.TERMS, dim=-1)], dim=-1)

  def __call__(self, p: Proprio) -> torch.Tensor:
    self.last = self.policy(self.obs(p))
    return self.home + self.scale * self.last


class HeadModule:
  """Ball-tracking head (Mjlab-HeadTrack): detection (visible, du, dv) + proprio -> HeadCmd.
  Obs = det 3 | head q-home 2 | head dq 2 | gyro 3 | last action 2; target = home + 0.5 a."""
  SCALE = 0.5

  def __init__(self, ckpt: str, num_envs: int = 1, device="cpu"):
    self.policy = PolicyModule(ckpt, device)
    self.home = torch.tensor(HOME[HEAD], device=device)
    self.last = torch.zeros(num_envs, 2, device=device)

  def reset(self, ids=slice(None)) -> None:
    self.last[ids] = 0.0

  def obs(self, p: Proprio, det: torch.Tensor) -> torch.Tensor:
    return torch.cat([det, p.q[:, HEAD] - self.home, p.dq[:, HEAD], p.ang_vel, self.last], -1)

  def __call__(self, p: Proprio, det: torch.Tensor) -> HeadCmd:
    self.last = self.policy(self.obs(p, det))
    return HeadCmd(self.home + self.SCALE * self.last)


#: arm pose the deploy holds while walking (L 4 | R 4), and the kinematic chain in the trunk frame (K1_22dof.urdf,
#: y offsets times side +1 left / -1 right): joint origin, axis index. Hand point = HAND_Y along the forearm y.
ARM_POSE = (-0.3, -1.25, 0.0, -1.4, -0.3, 1.25, 0.0, 1.4)
ARM_CHAIN = (((0.0, 0.077, 0.1845), 1), ((0.0025, 0.068, -0.0135), 0), ((0.0, 0.044428, 0.0), 1), ((0.0, 0.1215, 0.0), 2))
HAND_Y = 0.23


def _rot_batch(axis: int, a: torch.Tensor) -> torch.Tensor:
  c, s, o, i = a.cos(), a.sin(), torch.zeros_like(a), torch.ones_like(a)
  rows = {0: (i, o, o, o, c, -s, o, s, c), 1: (c, o, s, o, i, o, -s, o, c), 2: (c, -s, o, s, c, o, o, o, i)}[axis]
  return torch.stack(rows, -1).reshape(*a.shape, 3, 3)


def hand_fk(q: torch.Tensor, side: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """(n, 4) arm joints -> hand (n, 3) in the trunk frame, Jacobian (n, 3, 4), elbow (n, 3)."""
  n = q.shape[0]
  p, r = torch.zeros(n, 3, device=q.device, dtype=q.dtype), torch.eye(3, device=q.device, dtype=q.dtype).expand(n, 3, 3)
  pivots, axes = [], []
  flip = torch.tensor((1.0, side, 1.0), device=q.device, dtype=q.dtype)
  for k, (off, ax) in enumerate(ARM_CHAIN):
    p = p + (r @ (torch.tensor(off, device=q.device, dtype=q.dtype) * flip)[..., None])[..., 0]
    r = r @ _rot_batch(ax, q[:, k])
    pivots.append(p)
    axes.append(r[..., ax])
  hand = p + side * HAND_Y * r[..., 1]
  jac = torch.stack([torch.cross(w, hand - c, dim=-1) for w, c in zip(axes, pivots)], -1)
  return hand, jac, p


class ArmIK:
  """Batched position IK for both hands (damped least squares, integrated on the commanded pose, so a stretched arm
  does not wind up). Same chain, clearance test and limits as the deploy jog. step(goal) -> 8 arm joint targets."""
  DAMP, ERR_MAX = 0.1, 0.02  # m: high damping and a clamped task error, or a stretched arm limit-cycles at the rate limit

  def __init__(self, num_envs: int = 1, device="cpu", rate: float = 3.0, dt: float = 0.02):
    self.home = torch.tensor(ARM_POSE, device=device)
    self.lim = torch.tensor(LIMITS[ARMS], device=device)
    self.max_step, self.q = rate * dt, self.home.repeat(num_envs, 1)

  def reset(self, ids=slice(None)) -> None:
    self.q[ids] = self.home

  @staticmethod
  def clear(q: torch.Tensor, side: float) -> torch.Tensor:
    """(n,) bool: elbow to hand stays on its own side and outside the trunk box (the sim has no arm-trunk contact)."""
    hand, _, elbow = hand_fk(q, side)
    pts = torch.stack((elbow, (elbow + hand) / 2.0, hand), 1)
    y = side * pts[..., 1]
    bad = (y < 0.05) | ((pts[..., 0] < 0.11) & (y < 0.14) & (pts[..., 2] > -0.14))
    return ~bad.any(1)

  def step(self, goal: torch.Tensor) -> torch.Tensor:
    """goal (n, 6) = left hand xyz | right hand xyz, trunk frame."""
    for k, side in enumerate((1.0, -1.0)):
      idx = slice(4 * k, 4 * k + 4)
      q = self.q[:, idx]
      hand, j, _ = hand_fk(q, side)
      jt = j.transpose(1, 2)
      a = j @ jt + self.DAMP**2 * torch.eye(3, device=q.device, dtype=q.dtype)
      err = goal[:, 3 * k:3 * k + 3] - hand
      err = err * (self.ERR_MAX / err.norm(dim=-1, keepdim=True).clamp_min(self.ERR_MAX))
      dq = (jt @ torch.linalg.solve(a, err[..., None]))[..., 0]
      dq = dq * (self.max_step / dq.abs().amax(-1, keepdim=True).clamp_min(self.max_step))
      new = torch.maximum(torch.minimum(q + dq, self.lim[idx, 1]), self.lim[idx, 0])
      self.q[:, idx] = torch.where(self.clear(new, side)[:, None], new, q)
    return self.q


class DecoupledWBC:
  """One 50 Hz step of the body layer: commands + proprio -> 22 joint PD targets (model order).

  Upper body defaults to HOME when no command is given, so a task layer that only walks needs only LocoCmd."""

  def __init__(self, loco_ckpt: str, num_envs: int = 1, device="cpu"):
    self.loco = LocoModule(loco_ckpt, num_envs, device)
    self.home = torch.tensor(HOME, device=device).expand(num_envs, 22)

  def reset(self, ids=slice(None)) -> None:
    self.loco.reset(ids)

  def step(self, p: Proprio, loco: LocoCmd, head: HeadCmd | None = None, arms: ArmCmd | None = None) -> torch.Tensor:
    q = self.home.clone()
    q[:, LEGS] = self.loco(p, loco)
    if head is not None:
      q[:, HEAD] = head.tensor()
    if arms is not None:
      q[:, ARMS] = arms.tensor()
    return q
