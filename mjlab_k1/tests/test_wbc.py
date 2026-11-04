"""WP0 contract checks for wbc.py (loaded by path: no mjlab needed)."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("wbc", Path(__file__).parents[1] / "src/mjlab_k1/wbc.py")
wbc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wbc)


def test_sizes():
  assert len(wbc.JOINTS) == len(wbc.HOME) == len(wbc.LIMITS) == 22
  assert len(wbc.LEG_SCALE) == 12
  assert wbc.JOINTS[wbc.LEGS][0] == "Left_Hip_Pitch" and wbc.JOINTS[wbc.ARMS][0] == "ALeft_Shoulder_Pitch"


def test_mirror_involution_and_home():
  q = torch.randn(5, 22)
  assert torch.allclose(wbc.mirror_joints(wbc.mirror_joints(q)), q)
  home = torch.tensor(wbc.HOME)
  assert torch.allclose(wbc.mirror_joints(home), home)


def test_mirror_maps_limits():
  lim = torch.tensor(wbc.LIMITS)
  lo, hi = wbc.mirror_joints(lim[:, 0]), wbc.mirror_joints(lim[:, 1])
  m = torch.minimum(lo, hi), torch.maximum(lo, hi)
  assert torch.allclose(m[0], lim[:, 0]) and torch.allclose(m[1], lim[:, 1])


def test_command_clamps():
  big = torch.tensor([5.0])
  assert wbc.LocoCmd(big, -big, big).tensor().tolist() == [[1.0, -0.5, 1.0]]
  assert torch.allclose(wbc.HeadCmd(torch.tensor([[2.0, 2.0]])).tensor(), torch.tensor([[1.0, 0.855]]))
  arm = wbc.ArmCmd(torch.full((1, 8), 9.0)).tensor()
  assert torch.allclose(arm, torch.tensor(wbc.LIMITS[2:10])[:, 1][None])
  c = wbc.LocoCmd.from_action(torch.tensor([[0.5, 1.0, -2.0]]))
  assert c.tensor().tolist() == [[0.5, 0.5, -1.0]]


def test_loco_history_term_major(tmp_path):
  # identity-ish policy: 225 -> 12, zero weights so output = bias; checks obs shape and fresh-fill
  sd = {"mlp.0.weight": torch.zeros(12, 225), "mlp.0.bias": torch.ones(12),
        "obs_normalizer._mean": torch.zeros(225), "obs_normalizer._std": torch.ones(225)}
  ck = tmp_path / "m.pt"
  torch.save({"actor_state_dict": sd}, ck)
  w = wbc.DecoupledWBC(str(ck), num_envs=2)
  p = wbc.Proprio(torch.zeros(2, 3), torch.tensor([[0.0, 0.0, -1.0]] * 2), torch.tensor(wbc.HOME).repeat(2, 1),
                  torch.zeros(2, 22))
  cmd = wbc.LocoCmd(torch.tensor([0.3, 0.3]), torch.zeros(2), torch.zeros(2))
  obs = w.loco.obs(p, cmd)
  assert obs.shape == (2, 225)
  # term-major, oldest->newest inside a term; fresh reset fills all 5 steps with the first frame
  assert torch.allclose(obs[:, -15:], torch.tensor([0.3, 0.0, 0.0]).repeat(2, 5))
  assert torch.allclose(obs[:, :15], torch.zeros(2, 15))
  q = w.step(p, cmd)
  assert torch.allclose(q[:, wbc.LEGS], torch.tensor(wbc.HOME[10:]) + torch.tensor(wbc.LEG_SCALE))
  assert torch.allclose(q[:, wbc.ARMS], torch.tensor(wbc.HOME[2:10]).expand(2, 8))


def test_head_module_obs_and_clamp(tmp_path):
  sd = {"mlp.0.weight": torch.zeros(2, 12), "mlp.0.bias": torch.tensor([4.0, -4.0]),
        "obs_normalizer._mean": torch.zeros(12), "obs_normalizer._std": torch.ones(12)}
  ck = tmp_path / "h.pt"
  torch.save({"actor_state_dict": sd}, ck)
  h = wbc.HeadModule(str(ck))
  p = wbc.Proprio(torch.ones(1, 3), torch.tensor([[0.0, 0.0, -1.0]]), torch.tensor(wbc.HOME)[None], torch.zeros(1, 22))
  det = torch.tensor([[1.0, 0.2, -0.1]])
  assert torch.allclose(h.obs(p, det), torch.tensor([[1.0, 0.2, -0.1, 0, 0, 0, 0, 1, 1, 1, 0, 0]]))
  cmd = h(p, det)
  assert torch.allclose(h.last, torch.tensor([[4.0, -4.0]]))       # raw action kept for the next obs
  assert torch.allclose(cmd.tensor(), torch.tensor([[1.0, -0.349]]))  # target clamped to the head range
