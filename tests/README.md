# tests

Part of [k1-runtime](https://github.com/thdhyan/k1-runtime).

| File | What it does |
|---|---|
| `test_arm_ik.py` | wbc.ArmIK / hand_fk against MuJoCo FK on the K1 model. Needs mujoco + torch, no mjlab. |
| `test_gamepad_core.py` | k1_runtime.gamepad_core paths the MuJoCo sim-to-sim run does not exercise (moved from k1-deploy-robot). |
| `test_wbc.py` | WP0 contract checks for k1_runtime.wbc (no mjlab needed). |
