"""Shared half of the K1 gamepad deploy: gains, Walker, Body FSM, ArmJog, Gamepad (evdev / K1 remote / UDP / keyboard).

Split out of mjlab_k1/deploy/k1_gamepad_deploy.py. The robot runner (k1-deploy-robot) and the laptop sim-to-sim
(k1-deploy-laptop) both import this module; torch + numpy only.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from k1_runtime import wbc  # noqa: E402  (torch only, no mjlab import)

DT = 0.02
#: first-hardware command cap (vx, vy, wz); the base was trained to LocoCmd.RANGE
CAP = (0.5, 0.3, 0.8)
DEADZONE = 0.2  # DS4 sticks rest at up to 0.14
#: --udp: no laptop pad packet for this long (s) -> zero command + stop (B)
UDP_STALE = 1.0
#: --keyboard: a terminal has no key-release event, a key counts as held until this long (s) after its last repeat.
#: Must outlast the terminal's auto-repeat delay (usually 0.25-0.66 s).
KEY_HOLD = 0.7
#: --keyboard: keys 1 / 2 / 3 = this fraction of the --cap limits; each must stay above DEADZONE
KEY_LEVELS = {"1": 0.25, "2": 0.5, "3": 1.0}
KEY_HELP = """keyboard (hold to move, release = zero command within %.1f s):
  ENTER start   SPACE stop   Ctrl-C stop
  w/s forward/back   a/d left/right   q/e turn left/right   1/2/3 speed 25/50/100%% of cap (default 50%%)
  arrows head   r/f trunk up/down (height policies)   h height back to stand""" % KEY_HOLD
#: D-pad head speed, rad/s
HEAD_RATE = 0.8
#: right-stick trunk height speed, m/s
HEIGHT_RATE = 0.1
#: hand IK jog: hand speed m/s, joint speed limit rad/s, damped least squares lambda
HAND_RATE, ARM_RATE, IK_DAMP = 0.15, 1.5, 0.03
_W = 4.0 * 2.0 * math.pi


def _pd(armature, zeta=1.5, ratio=1.0):
  a = armature * ratio
  return a * _W**2, 2.0 * zeta * a * _W


_LEG = (_pd(0.0478125), _pd(0.0339552), _pd(0.0282528), _pd(0.095625, 1.0), _pd(0.0282528, ratio=1.4),
        _pd(0.0282528, ratio=0.4))
_UPPER = _pd(0.001, 1.0)
KP = np.array((_UPPER[0],) * 10 + tuple(k for k, _ in _LEG) * 2, dtype=np.float32)
KD = np.array((_UPPER[1],) * 10 + tuple(d for _, d in _LEG) * 2, dtype=np.float32)
#: head at booster_deploy's policy gains: the training kp 0.63 sags 0.7 rad under gravity, fine for the head
#: policy (it learned to compensate) but not for direct D-pad targets. Too light to disturb the walk base.
KP[:2], KD[:2] = 4.0, 1.0
#: arms held in front, elbows bent (L | R: shoulder pitch, roll, elbow pitch, elbow yaw = bend): at HOME (hanging,
#: kp 0.63) the arms sag 0.4-0.5 rad into the trunk. kp 20/10 holds 0.03 rad; walk_still_v1 tracking and
#: yaw unchanged in sim-to-sim for arm kp 0.63-40.
ARM_POSE = np.array((-0.3, -1.25, 0.0, -1.4, -0.3, 1.25, 0.0, 1.4), dtype=np.float32)
KP[2:10] = (20.0, 20.0, 10.0, 10.0) * 2
KD[2:10] = (1.0, 1.0, 0.5, 0.5) * 2
LIM = np.array(wbc.LIMITS, dtype=np.float32)
HOME = np.array(wbc.HOME, dtype=np.float32)
HOME[2:10] = ARM_POSE
#: stock Prepare-mode pose (straight legs), measured on A2 2026-10-28
PREPARE = np.array((0.0, 0.0, 0.19, -1.45, 0.0, -0.5, 0.19, 1.45, 0.0, 0.5)
                   + (0.0, 0.0, 0.0, 0.105, -0.10, 0.0) * 2, dtype=np.float32)
#: Custom entry: interpolate measured pose -> HOME before the policy starts (it only saw poses near HOME).
#: The training gains cannot hold a static pose without the policy balancing, so hold-prime and ramp use
#: booster_deploy's K1 prepare_state gains (booster_deploy/robots/k1.py), as booster_deploy does.
RAMP_S = 2.0
FADE_S = 2.0  # normal stop: policy on, zero walk, head / height / arms blend home over this, then stock Walking
PREP_KP = np.array((40, 40, 40, 50, 20, 20, 40, 50, 20, 20) + (350, 350, 180, 350, 250, 250) * 2, dtype=np.float32)
PREP_KD = np.array((1.5, 1.5, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2) + (7.5, 7.5, 3, 5.5, 5, 5) * 2,
                   dtype=np.float32)
#: get-up policy at its training gains (Mjlab-GetUp: head + legs 4 Hz PD, arms kp 40 / kd 2); DAMP = kp 0, same kd
GETUP_KP = np.array((_UPPER[0],) * 2 + (40.0,) * 8 + tuple(k for k, _ in _LEG) * 2, dtype=np.float32)
GETUP_KD = np.array((_UPPER[1],) * 2 + (2.0,) * 8 + tuple(d for _, d in _LEG) * 2, dtype=np.float32)
ZERO = np.zeros(22, dtype=np.float32)
#: FSM timing (s): limp before the get-up, get-up timeout, upright hold before hand-over, ramp to HOME
DAMP_S, GETUP_S, STOOD_S, RAMP_UP_S, GETUP_TRIES = 1.0, 10.0, 0.5, 1.0, 3


def gravity_from_rpy(r, p, _y=0.0):
  """Body-frame unit gravity for world R = Rz(y) Ry(p) Rx(r) (booster rpy, IsaacLab quat_from_euler_xyz)."""
  return np.array([math.sin(p), -math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r)], dtype=np.float32)


def stick_to_cmd(lx, ly, rx):
  dz = lambda v: 0.0 if abs(v) < DEADZONE else max(-1.0, min(1.0, v))  # noqa: E731
  return np.array([-dz(ly) * CAP[0], -dz(lx) * CAP[1], -dz(rx) * CAP[2]], dtype=np.float32)


#: arm chain in the trunk frame (K1_22dof.urdf, y offsets times side +1 left / -1 right): joint origin, axis index
ARM_CHAIN = (((0.0, 0.077, 0.1845), 1), ((0.0025, 0.068, -0.0135), 0), ((0.0, 0.044428, 0.0), 1), ((0.0, 0.1215, 0.0), 2))
#: hand point along the forearm link y, m (Left_Arm_4 mesh tip)
HAND_Y = 0.23


def _rot(axis, a):
  c, s = math.cos(a), math.sin(a)
  return {0: np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 1: np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
          2: np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def hand_fk(q, side):
  """Hand position in the trunk frame and its 3x4 Jacobian; q = shoulder pitch, roll, elbow pitch, elbow yaw."""
  p, r, pivots, axes = np.zeros(3), np.eye(3), [], []
  for (off, ax), a in zip(ARM_CHAIN, q):
    p = p + r @ (np.array(off) * (1.0, side, 1.0))
    r = r @ _rot(ax, a)
    pivots.append(p)
    axes.append(r[:, ax])
  hand = p + r @ np.array((0.0, side * HAND_Y, 0.0))
  return hand, np.stack([np.cross(w, hand - c) for w, c in zip(axes, pivots)], 1), p


def arm_clear(q, side):
  """Elbow to hand stays on its own side and out of the trunk mesh box + arm radius (MuJoCo has no arm-trunk contact)."""
  hand, _, elbow = hand_fk(q, side)
  for x, y, z in (elbow, (elbow + hand) / 2.0, hand):
    if side * y < 0.05 or (x < 0.11 and side * y < 0.14 and z > -0.14):
      return False
  return True


class ArmJog:
  """Velocity IK for both arms, integrated on the commanded pose so a stretched arm does not wind up."""
  SIDES = ((slice(0, 4), 1.0), (slice(4, 8), -1.0))

  def __init__(self):
    self.q, self.homing = ARM_POSE.astype(np.float64), False

  def step(self, dt, left, right, vel):
    """vel = left-hand velocity in the trunk frame; the right hand mirrors y. Returns the 8 arm targets."""
    if left or right:
      self.homing = False
    if self.homing:
      self.q += np.clip(ARM_POSE - self.q, -ARM_RATE * dt, ARM_RATE * dt)
      self.homing = bool(np.any(self.q != ARM_POSE))
    for (idx, side), held in zip(self.SIDES, (left, right)):
      if held and np.any(vel):
        j = hand_fk(self.q[idx], side)[1]
        dq = j.T @ np.linalg.solve(j @ j.T + IK_DAMP**2 * np.eye(3), np.asarray(vel) * (1.0, side, 1.0) * dt)
        q = np.clip(self.q[idx] + np.clip(dq, -ARM_RATE * dt, ARM_RATE * dt), LIM[2:10][idx, 0], LIM[2:10][idx, 1])
        if arm_clear(q, side):
          self.q[idx] = q
    return self.q.astype(np.float32)


class Walker:
  """50 Hz: (gyro, gravity, q, dq, cmd) in model order -> 22 clamped joint targets."""

  def __init__(self, ckpt):
    self.wbc = wbc.DecoupledWBC(ckpt)

  def reset(self):
    self.wbc.reset()

  def __call__(self, gyro, grav, q, dq, cmd, head=(0.0, 0.0), dh=0.0, arms=ARM_POSE):
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)[None]  # noqa: E731
    p = wbc.Proprio(t(gyro), t(grav), t(q), t(dq))
    c = t(cmd)[0]
    target = self.wbc.step(p, wbc.LocoCmd(c[:1], c[1:2], c[2:3], t(dh)), wbc.HeadCmd(t(head)),
                           wbc.ArmCmd(t(arms)))[0].numpy()
    return np.clip(target, LIM[:, 0], LIM[:, 1])


class Body:
  """22-joint command FSM: WALK -(fall / pad.getup)-> DAMP -> GETUP -(stood)-> RAMP -> WALK.
  auto: the get-up starts by itself after DAMP_S (sim); else on pad.getup (robot: spotter ready)."""

  def __init__(self, walker, getup_ckpt=None, auto=True):
    self.walker, self.auto = walker, auto
    self.getup = wbc.GetUpModule(getup_ckpt) if getup_ckpt else None
    self.home = np.array(wbc.HOME, dtype=np.float32)  # get-up hand-over pose (arms down)
    self.state, self.n, self.tries, self.stood, self.ramp = "WALK", 0, 0, 0, []

  def _to(self, state):
    self.state, self.n = state, 0
    print(f"\n-> {state}")

  def __call__(self, gyro, grav, q, dq, pad):
    """-> (target, kp, kd), or None = give up (fall without a get-up policy, or GETUP_TRIES failed)."""
    self.n += 1
    if self.state in ("WALK", "RAMP") and (grav[2] > -0.5 or pad.getup):
      if self.getup is None:
        return None
      pad.getup = False
      self._to("DAMP")
    if self.state == "WALK":
      return self.walker(gyro, grav, q, dq, pad.walk_cmd(), pad.head_step(DT), pad.height_step(DT),
                         pad.arm_step(DT)), KP, KD
    if self.state == "RAMP":
      target = self.ramp.pop(0)
      if not self.ramp:
        self.walker.reset()
        pad.arm_reset()
        self.tries = 0
        self._to("WALK")
      return target, PREP_KP, PREP_KD
    if self.state == "DAMP":
      if self.n * DT < DAMP_S or not (self.auto or pad.getup):
        return q, ZERO, GETUP_KD
      if self.tries >= GETUP_TRIES:
        return None
      pad.getup, self.stood = False, 0
      self.tries += 1
      self.getup.reset()
      self._to("GETUP")
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]  # noqa: E731
    target = np.clip(self.getup(wbc.Proprio(t(gyro), t(grav), t(q), t(dq)))[0].numpy(), LIM[:, 0], LIM[:, 1])
    # training "stood" = trunk > 0.45 m and up > 0.9; the robot has no height, so legs near HOME + still instead
    up = grav[2] < -0.9 and np.abs(q - self.home)[10:].max() < 0.35 and np.linalg.norm(gyro) < 0.5
    self.stood = self.stood + 1 if up else 0
    if self.stood * DT >= STOOD_S:
      self.ramp = list(np.linspace(q, HOME, int(RAMP_UP_S / DT)))
      self._to("RAMP")
    elif self.n * DT > GETUP_S:
      self._to("DAMP")
    return target, GETUP_KP, GETUP_KD


class Still:
  """Pad stand-in for scripted sim runs: zero command, no buttons."""
  cmd, getup, knock = np.zeros(3, np.float32), False, False

  def head_step(self, dt):
    return (0.0, 0.0)

  def height_step(self, dt):
    return 0.0

  def walk_cmd(self):
    return self.cmd

  def arm_step(self, dt):
    return ARM_POSE

  def arm_reset(self):
    pass


# ---------------------------------------------------------------- MuJoCo sim-to-sim

#: scripted get-up test: (name, roll, pitch) of the trunk at drop
FALLS = (("prone", 0.0, 1.57), ("supine", 0.0, -1.57), ("left side", 1.57, 0.0), ("right side", -1.57, 0.0))

SCRIPT = (("stand", 2.0, (0, 0, 0)), ("fwd 0.5", 4.0, (0.5, 0, 0)), ("turn 0.8", 3.0, (0, 0, 0.8)),
          ("left 0.3", 3.0, (0, 0.3, 0)), ("back 0.3", 3.0, (-0.3, 0, 0)), ("fwd+turn", 4.0, (0.4, 0, -0.5)),
          ("stop", 2.0, (0, 0, 0)))


def rough_heights(size=12.0, res=0.05, seed=0):
  """Blind-walk test field (m, rows = y) around a flat 3 m start pad: random rough +x (0-4 cm, 0.2 m cells),
  pyramid stairs -x (6 cm x 0.3 m steps), pyramid slope +y (14 deg), waves -y."""
  n = int(2 * size / res) + 1
  x, y = np.meshgrid(np.linspace(-size, size, n), np.linspace(-size, size, n))
  rough = np.kron(np.random.default_rng(seed).uniform(0, 0.04, (n // 4 + 1,) * 2), np.ones((4, 4)))[:n, :n]
  stairs = np.clip((3.0 - np.maximum(abs(x + 6), abs(y))) // 0.3, 0, 8) * 0.06
  slope = np.clip((4.0 - np.maximum(abs(x), abs(y - 8))) * 0.25, 0, 0.75)
  waves = 0.025 * (2 + np.sin(2 * np.pi * x / 1.2) + np.sin(2 * np.pi * y / 1.2)) * np.clip(-y - 3, 0, 1)
  band = abs(y) < 3
  h = np.where((x > 1.5) & band, rough, 0.0)
  h = np.where((x < -1.5) & band, stairs, h)
  h = np.where((y > 3) & (abs(x) < 4), slope, h)
  return np.where((y < -3) & (abs(x) < 4), waves, h), size



# ---------------------------------------------------------------- robot

class Gamepad:
  """Latest stick command + A/B edges, from an evdev pad if one is plugged in, else the K1 remote topic."""

  def __init__(self, node, udp_port=None, keyboard=False):
    self.cmd, self.start, self.stop = np.zeros(3, np.float32), False, False
    self.getup, self.knock = False, False  # Y/Triangle: get up now; X/Square: push the sim robot over
    self.hat, self.head = np.zeros(2), np.zeros(2)  # D-pad (x, y); head target (yaw, pitch)
    self.ry, self.dh = 0.0, 0.0  # right stick Y; trunk height offset (m)
    self.lx, self.ly, self.l1, self.r1, self.jog = 0.0, 0.0, False, False, ArmJog()  # hand IK mode
    self.arms_used, self.dh_used = np.array(ARM_POSE, np.float32), 0.0  # last arms / height the policy got (fade start)
    self.task = None  # push_ik_phasespace.py: {"cmd": 3, "dh": m, "arms": 8 or None} replaces the sticks
    self.sock, self.peer, self.gap = None, None, 0.0  # --udp: status replies go to whoever sends the commands
    self.hangup = False  # --keyboard: Ctrl-C or the terminal is gone, exit
    if udp_port or keyboard:
      if keyboard:
        if not sys.stdin.isatty():
          raise SystemExit("--keyboard needs a terminal: use ssh -t, not nohup or a pipe")
        self._keyboard_start()
        print(KEY_HELP)
      else:
        threading.Thread(target=self._udp, args=(udp_port,), daemon=True).start()
        print(f"gamepad: laptop pad over udp :{udp_port} (+ K1 remote B = stop)")
      if node is not None:
        from booster_interface.msg import RemoteControllerState
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        node.create_subscription(RemoteControllerState, "/remote_controller_state",
                                 lambda m: setattr(self, "stop", self.stop or bool(m.b)),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
      return
    try:
      import evdev
      # a gamepad has sticks AND a south/A button (touchpads and gpio-keys have one or the other)
      dev = next(dev for dev in map(evdev.InputDevice, evdev.list_devices())
                 if evdev.ecodes.EV_ABS in dev.capabilities()
                 and evdev.ecodes.BTN_SOUTH in dev.capabilities().get(evdev.ecodes.EV_KEY, []))
      threading.Thread(target=self._evdev, args=(evdev, dev), daemon=True).start()
      print("gamepad:", dev.name)
    except (ImportError, StopIteration):
      if node is None:
        raise SystemExit("no gamepad found (need python-evdev and read access to /dev/input/event*)")
      from booster_interface.msg import RemoteControllerState
      from rclpy.qos import QoSProfile, ReliabilityPolicy
      node.create_subscription(RemoteControllerState, "/remote_controller_state", self._remote,
                               QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
      print("gamepad: K1 remote (/remote_controller_state)")

  def head_step(self, dt):
    """Integrate the held D-pad into the head target (right = yaw right, up = look up), clamped to limits."""
    self.head = np.clip(self.head + HEAD_RATE * dt * np.array([-self.hat[0], self.hat[1]]), LIM[:2, 0], LIM[:2, 1])
    return self.head

  def walk_cmd(self):
    if self.task:
      return np.clip(np.array(self.task["cmd"], np.float32), -np.array(CAP, np.float32), np.array(CAP, np.float32))
    return np.zeros(3, np.float32) if self.l1 or self.r1 else self.cmd

  def arm_step(self, dt):
    """Stick -> left-hand velocity (up = forward, left = left, right stick up = up) while R1 / L1 is held."""
    if self.task and self.task["arms"] is not None:
      self.arms_used = np.clip(np.array(self.task["arms"], np.float32), LIM[2:10, 0], LIM[2:10, 1])
      return self.arms_used
    dz = lambda v: 0.0 if abs(v) < DEADZONE else v  # noqa: E731
    self.arms_used = self.jog.step(dt, self.l1, self.r1, HAND_RATE * np.array([-dz(self.ly), -dz(self.lx), -dz(self.ry)]))
    return self.arms_used

  def arm_reset(self):
    self.jog = ArmJog()

  def height_step(self, dt):
    """Integrate the right stick Y into the trunk height offset (up = higher), clamped to [H_MIN - H_STAND, 0]."""
    if self.task:
      self.dh_used = float(np.clip(self.task["dh"], wbc.H_MIN - wbc.H_STAND, 0.0))
      return self.dh_used
    ry = 0.0 if abs(self.ry) < DEADZONE or self.l1 or self.r1 else self.ry
    self.dh = self.dh_used = float(np.clip(self.dh - HEIGHT_RATE * dt * ry, wbc.H_MIN - wbc.H_STAND, 0.0))
    return self.dh

  def _remote(self, msg):
    self.cmd = stick_to_cmd(msg.lx, msg.ly, msg.rx)
    self.start |= msg.a
    self.stop |= msg.b
    self.getup |= bool(getattr(msg, "y", False))

  def _udp(self, port):
    import json
    import socket
    sock = self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", port))
    sock.settimeout(0.1)
    last, t = None, None
    while True:
      try:
        data, self.peer = sock.recvfrom(1024)
        m = json.loads(data)
      except socket.timeout:
        if t is not None and time.monotonic() - t > UDP_STALE:
          self.cmd, self.hat, self.ry = np.zeros(3, np.float32), np.zeros(2), 0.0
          self.lx, self.ly, self.l1, self.r1 = 0.0, 0.0, False, False
          self.task = None
          self.stop = True
        continue
      self.gap = max(self.gap, time.monotonic() - t) if t is not None else 0.0
      t = time.monotonic()
      self.cmd = stick_to_cmd(m["lx"], m["ly"], m["rx"])
      self.task = m.get("task")
      self.hat, self.ry = np.array([m["hx"], m["hy"]], float), m["ry"]
      self.lx, self.ly = m["lx"], m["ly"]
      self.l1, self.r1 = bool(m.get("l1", 0)), bool(m.get("r1", 0))  # .get: an old sender must not kill this thread
      if last is not None:  # press counters: count up = pressed since the last packet
        self.start |= m["a"] > last["a"]
        self.stop |= m["b"] > last["b"]
        self.getup |= m["y"] > last["y"]
        self.knock |= m["x"] > last["x"]
        if m["r3"] > last["r3"]:
          self.dh = 0.0
        if m.get("l3", 0) > last.get("l3", 0):
          self.jog.homing = True
      last = m

  def _keyboard_start(self):
    import atexit
    import signal
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    def restore():
      try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
      except termios.error:  # terminal already gone
        pass
    atexit.register(restore)
    tty.setcbreak(fd)  # keys arrive at once, no echo
    # Ctrl-C arrives as a key: as a signal rclpy's handler shuts the context down, and ChangeMode back to Walking needs it
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~termios.ISIG
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)  # keys typed before this must not count
    # SSH drop: without this SIGHUP kills the process mid-policy, with no return to stock Walking. _keyboard stops instead.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    threading.Thread(target=self._keyboard, args=(fd,), daemon=True).start()

  def _keyboard(self, fd):
    import os
    import re
    import select
    level, held = KEY_LEVELS["2"], {}
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}
    while True:
      if select.select([fd], [], [], 0.02)[0]:
        try:
          data = os.read(fd, 256)
        except OSError:
          data = b""
        if not data or b"\x03" in data:  # terminal gone or Ctrl-C: leave. Repeat the stop, run_robot clears it once after ChangeMode.
          if not data:
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)  # a print to the dead terminal would raise and skip the mode change
            os.dup2(null, 2)
          self.hangup = True
          self.cmd, self.hat, self.ry = np.zeros(3, np.float32), np.zeros(2), 0.0
          while True:
            self.stop = True
            time.sleep(0.1)
        now = time.monotonic()
        for arrow, ch in re.findall(r"\x1b[\[O]([ABCD])|(.)", data.decode(errors="ignore"), re.S):
          if arrow:
            held[arrows[arrow]] = now
          elif (ch := ch.lower()) in KEY_LEVELS:
            level = KEY_LEVELS[ch]
          elif ch in "\r\n":
            self.start = True
          elif ch == " ":
            self.stop = True
          elif ch == "h":
            self.dh = 0.0
          elif ch in "wsadqerf":
            held[ch] = now
      now = time.monotonic()
      held = {k: t for k, t in held.items() if now - t < KEY_HOLD}
      axis = lambda neg, pos: float(pos in held) - float(neg in held)  # noqa: E731
      self.cmd = stick_to_cmd(level * axis("a", "d"), level * axis("w", "s"), level * axis("q", "e"))
      self.hat, self.ry = np.array([axis("left", "right"), axis("up", "down")]), axis("r", "f")

  def reply(self, status):
    """Tell the laptop this robot's state, with the longest command packet gap since the last reply."""
    if self.sock and self.peer:
      import json
      gap, self.gap = self.gap, 0.0
      self.sock.sendto(json.dumps({**status, "gap_ms": round(gap * 1e3)}).encode(), self.peer)

  def _evdev(self, evdev, dev):
    e = evdev.ecodes
    yaw = e.ABS_RX if e.ABS_RX in dict(dev.capabilities()[e.EV_ABS]) else e.ABS_Z
    axes = {e.ABS_X: 0.0, e.ABS_Y: 0.0, yaw: 0.0}
    pitch_axis = e.ABS_RY if yaw == e.ABS_RX else e.ABS_RZ
    for ev in dev.read_loop():
      if ev.type == e.EV_ABS and ev.code == pitch_axis:
        info = dev.absinfo(ev.code)
        self.ry = 2.0 * (ev.value - info.min) / (info.max - info.min) - 1.0
      elif ev.type == e.EV_ABS and ev.code in axes:
        info = dev.absinfo(ev.code)
        axes[ev.code] = 2.0 * (ev.value - info.min) / (info.max - info.min) - 1.0
        self.cmd = stick_to_cmd(axes[e.ABS_X], axes[e.ABS_Y], axes[yaw])
        self.lx, self.ly = axes[e.ABS_X], axes[e.ABS_Y]
      elif ev.type == e.EV_ABS and ev.code in (e.ABS_HAT0X, e.ABS_HAT0Y):
        self.hat[int(ev.code == e.ABS_HAT0Y)] = ev.value
      elif ev.type == e.EV_KEY and ev.code in (e.BTN_TL, e.BTN_TR):
        if ev.value != 2:  # 2 = autorepeat
          setattr(self, "l1" if ev.code == e.BTN_TL else "r1", ev.value == 1)
      elif ev.type == e.EV_KEY and ev.value == 1:
        if ev.code == e.BTN_THUMBL:
          self.jog.homing = True
        self.start |= ev.code in (e.BTN_A, e.BTN_SOUTH)
        self.stop |= ev.code in (e.BTN_B, e.BTN_EAST)
        self.getup |= ev.code == e.BTN_NORTH
        self.knock |= ev.code == e.BTN_WEST
        if ev.code == e.BTN_THUMBR:
          self.dh = 0.0
