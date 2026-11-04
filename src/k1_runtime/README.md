# src/k1_runtime

Part of [k1-runtime](https://github.com/thdhyan/k1-runtime).

| File | What it does |
|---|---|
| `ds4.py` | DS4 on the laptop for push_ik_web.py: sticks, buttons with press and hold, light bar. Wired or Bluetooth. |
| `gamepad_core.py` | Shared half of the K1 gamepad deploy: gains, Walker, Body FSM, ArmJog, Gamepad (evdev / K1 remote / UDP / keyboard). |
| `k1_arm_ik.py` | K1 arm IK (trunk frame) for VR teleop: damped least squares on the K1 URDF (``k1_teleop/K1_22dof-ZED.urdf``, a copy of booster_assets ``robo |
| `push_ik_core.py` | Shared-actor controller for Mjlab-PushIK-Booster-K1, without ROS or mjlab: mocap poses -> per-robot commands. |
| `wbc.py` | Decoupled whole-body controller (HRL WP0 contract + runtime), torch only: no mjlab/Isaac import. |
