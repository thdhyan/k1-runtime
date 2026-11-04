"""K1 arm IK (trunk frame) for VR teleop: damped least squares on the K1 URDF
(``k1_teleop/K1_22dof-ZED.urdf``, a copy of booster_assets
``robots/K1/K1_22dof-ZED.urdf``; meshes are not needed), warm-started from the
last solution. Also full-body FK for the headset's robot-pose panel.

K1 arms have 4 DoF and no wrist, so only the hand *position* is tracked:
``Shoulder_Pitch`` (y), ``Shoulder_Roll`` (x), ``Elbow_Pitch`` (about the upper
arm: a twist) and ``Elbow_Yaw`` (at ``*_hand_link``: the actual elbow bend). Joint names are the URDF's,
sort prefixes included (``ALeft_Shoulder_Pitch``, ``AAHead_yaw``).
The end effector is ``HAND_TIP`` along the forearm from ``*_hand_link``
(Left_Arm_4.STL reaches 0.228 m along +y).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIDES = {"left": "Left", "right": "Right"}
ARM = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")
HAND_TIP = 0.20            # m along the forearm (+y left, -y right) from *_hand_link
ELBOW_SEED = 0.15         # rad, see ArmIK.solve
ROOT = "Trunk"
URDF = Path(__file__).resolve().parent / "K1_22dof-ZED.urdf"


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def load_urdf_joints(path: Path) -> dict[str, dict]:
    """name -> {parent, child, type, origin (4x4), axis, lower, upper}."""
    joints = {}
    for node in ET.parse(path).getroot().findall("joint"):
        parent, child, origin = node.find("parent"), node.find("child"), node.find("origin")
        if parent is None or child is None:
            continue
        T = np.eye(4)
        if origin is not None:
            T[:3, :3] = _rpy(*(float(v) for v in origin.attrib.get("rpy", "0 0 0").split()))
            T[:3, 3] = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        axis_node, limit = node.find("axis"), node.find("limit")
        axis = np.array([float(v) for v in axis_node.attrib["xyz"].split()]) if axis_node is not None else np.array([1.0, 0, 0])
        joints[node.attrib["name"]] = {
            "parent": parent.attrib["link"], "child": child.attrib["link"], "type": node.attrib.get("type", "fixed"),
            "origin": T, "axis": axis / max(np.linalg.norm(axis), 1e-12),
            "lower": float(limit.attrib.get("lower", -math.pi)) if limit is not None else -math.pi,
            "upper": float(limit.attrib.get("upper", math.pi)) if limit is not None else math.pi,
        }
    return joints


class ArmIK:
    def __init__(self, urdf: Path = URDF):
        self.joints = load_urdf_joints(urdf)
        self._by_child = {j["child"]: (n, j) for n, j in self.joints.items()}
        # the URDF prefixes the first joint of each chain ("A", "AA") to fix the import order
        self.names = {s: [f"A{P}_{ARM[0]}"] + [f"{P}_{j}" for j in ARM[1:]] for s, P in SIDES.items()}
        self._chain = {s: self._chain_to(f"{s}_hand_link") for s in SIDES}
        self._tip = {}
        for s, sign in (("left", 1.0), ("right", -1.0)):
            self._tip[s] = np.eye(4)
            self._tip[s][1, 3] = sign * HAND_TIP
        self._lim = {s: np.array([[self.joints[n]["lower"], self.joints[n]["upper"]] for n in self.names[s]])
                     for s in SIDES}
        # shoulder centre: the roll joint, which also sits on the pitch axis
        self._shoulder_chain = {s: self._chain_to(f"{P}_Arm_2") for s, P in SIDES.items()}

    def _chain_to(self, link: str) -> list[tuple[str, dict]]:
        chain = []
        while link != ROOT:
            name, j = self._by_child[link]
            chain.append((name, j))
            link = j["parent"]
        return chain[::-1]

    @staticmethod
    def _chain_fk(chain, values: dict[str, float]) -> np.ndarray:
        T = np.eye(4)
        for name, j in chain:
            T = T @ j["origin"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], values.get(name, 0.0))
                T = T @ R
        return T

    def link_poses(self, state: dict[str, float]) -> dict[str, np.ndarray]:
        """Every URDF link's pose in the trunk frame for this joint state."""
        poses, todo = {ROOT: np.eye(4)}, list(self.joints.items())
        while todo:
            rest = []
            for name, j in todo:
                if j["parent"] not in poses:
                    rest.append((name, j))
                    continue
                T = poses[j["parent"]] @ j["origin"]
                if j["type"] in ("revolute", "continuous"):
                    R = np.eye(4)
                    R[:3, :3] = _axis_angle(j["axis"], state.get(name, 0.0))
                    T = T @ R
                poses[j["child"]] = T
            if len(rest) == len(todo):
                break                                   # links not under the trunk
            todo = rest
        for s in SIDES:                                 # hand tips, for drawing
            poses[f"{s}_hand_tip"] = poses[f"{s}_hand_link"] @ self._tip[s]
        return poses

    def skeleton(self, poses: dict[str, np.ndarray]) -> np.ndarray:
        """(M, 2, 3) parent->child link-origin segments, for drawing the robot."""
        segs = [(poses[j["parent"]][:3, 3], poses[j["child"]][:3, 3]) for j in self.joints.values()
                if j["parent"] in poses and j["child"] in poses]
        segs += [(poses[f"{s}_hand_link"][:3, 3], poses[f"{s}_hand_tip"][:3, 3]) for s in SIDES]
        return np.array(segs) if segs else np.zeros((0, 2, 3))

    def fk(self, side: str, state: dict[str, float], q: np.ndarray | None = None) -> np.ndarray:
        """Hand-tip pose (trunk frame)."""
        values = dict(state)
        if q is not None:
            values.update(zip(self.names[side], (float(v) for v in q)))
        return self._chain_fk(self._chain[side], values) @ self._tip[side]

    def arm_length(self, side: str) -> float:
        """Shoulder -> elbow -> hand tip (straight-arm reach), m."""
        p = self.link_poses({})
        P = SIDES[side]
        sh, el, tip = p[f"{P}_Arm_2"][:3, 3], p[f"{side}_hand_link"][:3, 3], p[f"{side}_hand_tip"][:3, 3]
        return float(np.linalg.norm(el - sh) + np.linalg.norm(tip - el))

    def shoulder_pos(self, side: str, state: dict[str, float]) -> np.ndarray:
        return self._chain_fk(self._shoulder_chain[side], state)[:3, 3]

    def solve(self, side: str, goal: np.ndarray, state: dict[str, float], iters: int = 30,
              damping: float = 0.02) -> tuple[np.ndarray, float]:
        """Joint targets putting the hand tip at ``goal[:3, 3]`` (4x4, trunk frame;
        the orientation is ignored); returns (q, position error m). ``state``
        supplies the warm start."""
        q = np.array([state.get(n, 0.0) for n in self.names[side]], dtype=np.float64)
        lo, hi = self._lim[side].T
        q = np.clip(q, lo, hi)
        # a straight arm (elbow at its 0 limit) has no gradient towards the shoulder:
        # start slightly bent; the solve straightens it again when the goal needs full reach
        bend = math.copysign(ELBOW_SEED, lo[3] + hi[3])
        if abs(q[3]) < ELBOW_SEED:
            q[3] = bend
        target = goal[:3, 3]
        best_q, best = q.copy(), float("inf")
        for _ in range(iters):
            p = self.fk(side, state, q)[:3, 3]
            e = target - p
            err = float(np.linalg.norm(e))
            if err < best:
                best, best_q = err, q.copy()
            if err < 1e-3:
                break
            J = np.zeros((3, 4))
            for i in range(4):
                dq = q.copy()
                dq[i] += 1e-5
                J[:, i] = (self.fk(side, state, dq)[:3, 3] - p) / 1e-5
            step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), e)
            q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
        return best_q, best
