"""wbc.ArmIK / hand_fk against MuJoCo FK on the K1 model. Needs mujoco + torch, no mjlab."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from k1_runtime import wbc  # noqa: E402

MODEL = ROOT / "src/k1_description/assets/robots/K1/K1_22dof.xml"
LEFT, RIGHT = wbc.JOINTS[2:6], wbc.JOINTS[6:10]


def _mj_hand(m, d, names, q, body):
  mujoco.mj_resetData(m, d)
  for n, v in zip(names, q):
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
  mujoco.mj_kinematics(m, d)
  t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
  return d.xmat[t].reshape(3, 3).T @ (d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)] - d.xpos[t])


def main():
  m = mujoco.MjModel.from_xml_path(str(MODEL))
  d = mujoco.MjData(m)
  torch.manual_seed(0)
  lim = torch.tensor(wbc.LIMITS[wbc.ARMS])
  print("hand body offset from the chain end, joints at 0:", _mj_hand(m, d, LEFT, [0] * 4, "left_hand_link"))
  # 1. FK of the 4-joint chain: the chain end (3rd return) is the hand_link origin; the hand point is HAND_Y past it
  q = lim[:, 0] + (lim[:, 1] - lim[:, 0]) * torch.rand(200, 8)
  worst = 0.0
  for k, (names, body, side) in enumerate(((LEFT, "left_hand_link", 1.0), (RIGHT, "right_hand_link", -1.0))):
    qs = q[:, 4 * k:4 * k + 4].double()
    h, j, end = wbc.hand_fk(qs, side)
    mj = np.array([_mj_hand(m, d, names, x.numpy(), body) for x in qs])
    err = np.linalg.norm(end.numpy() - mj, axis=1)
    worst = max(worst, err.max())
    print(f"side {side:+}: FK error vs {body}: mean {err.mean()*1000:.2f} mm, max {err.max()*1000:.2f} mm")
    # 2. Jacobian against finite differences
    eps = 1e-6
    for c in range(4):
      dq = qs.clone(); dq[:, c] += eps
      num = (wbc.hand_fk(dq, side)[0] - h) / eps
      assert (num - j[:, :, c]).abs().max() < 1e-4, c
  # 3. IK reaches the policy's hand-goal box centre region and stays legal
  ik = wbc.ArmIK(4, rate=3.0)
  goal = torch.tensor([[0.27, 0.2, -0.1, 0.27, -0.2, -0.1], [0.33, 0.15, 0.0, 0.33, -0.15, 0.0],
                       [0.2, 0.2, -0.15, 0.2, -0.2, -0.15], [0.35, 0.2, 0.1, 0.35, -0.2, 0.1]])
  for _ in range(200):
    out = ik.step(goal)
  reached = torch.cat([wbc.hand_fk(out[:, :4], 1.0)[0], wbc.hand_fk(out[:, 4:], -1.0)[0]], -1)
  err = (reached - goal).reshape(-1, 2, 3).norm(dim=-1)
  print("IK error per goal (m):", err.numpy().round(3).tolist())
  assert (out >= lim[:, 0] - 1e-6).all() and (out <= lim[:, 1] + 1e-6).all()
  assert worst < 0.02, f"FK off by {worst} m"
  # 4. no limit cycle anywhere in the policy's hand-goal box, reachable or not
  c, sc, f = torch.tensor((0.27, 0.2, -0.1)), torch.tensor((0.13, 0.12, 0.2)), torch.tensor((1.0, -1.0, 1.0))
  ik = wbc.ArmIK(2000)
  far = torch.cat([c, c * f]) + torch.cat([sc, sc * f]) * (2 * torch.rand(2000, 6) - 1)
  for _ in range(500):
    prev = ik.q.clone()
    ik.step(far)
  print("steady max |dq| per step:", (ik.q - prev).abs().max().item())
  assert (ik.q - prev).abs().max() < 1e-2
  assert err.max() < 0.03, err
  print("OK")


if __name__ == "__main__":
  main()
