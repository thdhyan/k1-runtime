"""k1_runtime.gamepad_core paths the MuJoCo sim-to-sim run does not exercise (moved from k1-deploy-robot)."""
import numpy as np

from k1_runtime import gamepad_core as k


def _rot(r, p, y):
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def test_gravity_from_rpy_matches_rotation():
  rng = np.random.default_rng(0)
  for r, p, y in rng.uniform(-1.0, 1.0, (50, 3)):
    want = _rot(r, p, y).T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(k.gravity_from_rpy(r, p, y), want, atol=1e-6)


def test_stick_mapping_signs_deadzone_cap():
  # K1 remote / evdev: stick up = negative Y, left = negative X (booster_deploy remote_control_service)
  np.testing.assert_allclose(k.stick_to_cmd(0.0, -1.0, 0.0), [k.CAP[0], 0, 0])
  np.testing.assert_allclose(k.stick_to_cmd(-1.0, 0.0, 0.0), [0, k.CAP[1], 0])
  np.testing.assert_allclose(k.stick_to_cmd(0.0, 0.0, -1.0), [0, 0, k.CAP[2]])
  np.testing.assert_allclose(k.stick_to_cmd(0.05, -0.05, 0.09), [0, 0, 0])
  np.testing.assert_allclose(k.stick_to_cmd(0.0, -3.0, 0.0), [k.CAP[0], 0, 0])


def test_gains_match_exported_mjcf():
  import re
  import k1_description
  xml = k1_description.robot_file(k1_description.MJCF_MJLAB).read_text()
  gains = {m[0]: (float(m[1]), -float(m[2])) for m in
           re.findall(r'<general name="(\w+)".*?gainprm="([\d.]+)" biasprm="0 -[\d.]+ (-[\d.]+)"', xml)}
  for i, j in list(enumerate(k.wbc.JOINTS))[10:]:  # head (D-pad) and arms (ARM_POSE) deliberately stiffer
    np.testing.assert_allclose((k.KP[i], k.KD[i]), gains[j], rtol=1e-4)


def test_getup_fsm_walk_damp_getup_ramp_walk():
  import torch

  class Stub:
    def __init__(self, out):
      self.out, self.resets = out, 0

    def reset(self):
      self.resets += 1

    def __call__(self, *a):
      return self.out

  home = np.array(k.wbc.HOME, np.float32)
  walker = Stub(k.HOME)
  body = k.Body(walker, auto=False)
  body.getup = Stub(torch.as_tensor(home)[None])
  pad, up, down, z = k.Still(), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.zeros(22)
  assert body(z[:3], up, k.HOME, z, pad)[1] is k.KP and body.state == "WALK"
  body(z[:3], down, home, z, pad)  # fallen
  assert body.state == "DAMP"
  for _ in range(int(2 * k.DAMP_S / k.DT)):  # robot: waits for Y however long it lies there
    target, kp, kd = body(z[:3], down, home, z, pad)
  assert body.state == "DAMP" and not kp.any()
  pad.getup = True
  assert body(z[:3], down, home, z, pad)[1] is k.GETUP_KP and body.state == "GETUP"
  for _ in range(int(k.STOOD_S / k.DT)):  # upright at HOME, still -> hand over
    body(z[:3], up, home, z, pad)
  assert body.state == "RAMP"
  for _ in range(int(k.RAMP_UP_S / k.DT)):
    target, kp, _ = body(z[:3], up, home, z, pad)
  assert body.state == "WALK" and kp is k.PREP_KP and walker.resets == 1
  np.testing.assert_allclose(target, k.HOME)


def test_getup_fsm_without_policy_gives_up_on_fall():
  body = k.Body(lambda *a: k.HOME)
  assert body(np.zeros(3), np.array([1.0, 0, 0]), k.HOME, np.zeros(22), k.Still()) is None


def test_hand_jacobian_matches_finite_difference():
  rng = np.random.default_rng(0)
  for side in (1.0, -1.0):
    for q in rng.uniform(-1.0, 1.0, (10, 4)):
      j = k.hand_fk(q, side)[1]
      num = np.stack([(k.hand_fk(q + 1e-6 * e, side)[0] - k.hand_fk(q - 1e-6 * e, side)[0]) / 2e-6 for e in np.eye(4)], 1)
      np.testing.assert_allclose(j, num, atol=1e-6)


def test_arm_jog_moves_held_hand_only_and_mirrors():
  jog = k.ArmJog()
  l0, r0 = k.hand_fk(jog.q[:4], 1.0)[0], k.hand_fk(jog.q[4:], -1.0)[0]
  for _ in range(25):  # 0.5 s at 0.1 m/s = 5 cm
    jog.step(k.DT, True, False, (0.1, 0.0, 0.0))
  np.testing.assert_allclose(k.hand_fk(jog.q[:4], 1.0)[0] - l0, [0.05, 0, 0], atol=6e-3)
  np.testing.assert_allclose(k.hand_fk(jog.q[4:], -1.0)[0], r0)
  jog = k.ArmJog()
  for _ in range(25):
    jog.step(k.DT, True, True, (0.0, 0.1, 0.0))
  l1, r1 = k.hand_fk(jog.q[:4], 1.0)[0], k.hand_fk(jog.q[4:], -1.0)[0]
  np.testing.assert_allclose((l1 - l0)[1], 0.05, atol=6e-3)
  np.testing.assert_allclose((r1 - r0)[1], -0.05, atol=6e-3)


def test_arm_jog_stays_in_limits_and_keep_out_then_homes():
  rng = np.random.default_rng(0)
  jog = k.ArmJog()
  for _ in range(40):
    v = rng.uniform(-0.15, 0.15, 3)
    for _ in range(50):
      q = jog.step(k.DT, True, True, v)
      assert np.all(q >= k.LIM[2:10, 0] - 1e-5) and np.all(q <= k.LIM[2:10, 1] + 1e-5)
      assert k.arm_clear(q[:4], 1.0) and k.arm_clear(q[4:], -1.0)
  jog.homing = True
  for _ in range(500):
    q = jog.step(k.DT, False, False, (0.0, 0.0, 0.0))
  np.testing.assert_allclose(q, k.ARM_POSE)
  assert not jog.homing
