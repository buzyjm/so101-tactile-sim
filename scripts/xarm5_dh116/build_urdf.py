"""Assemble one URDF for xArm5 + flange adapter + Leadshine DH116 right hand.

The two vendor packages are copied into ``assets/xarm5_dh116/src`` so the
merged URDF only uses relative mesh paths, and a small frustum STL is
generated to bridge the xArm5 flange (r = 37.8 mm) to the DH116 base
(r = 24.6 mm).  Run once, then convert with ``convert_assets.py``.

Inputs (download by hand, see README):
  --xarm-urdf   plain xarm5 URDF expanded from xarm_ros' xacro
  --xarm-meshes xarm_ros/xarm_description/meshes
  --dh116-pkg   DH116-R000-A1 folder from the current vendor package
"""
from __future__ import annotations

import argparse
import copy
import math
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dh116_joint_model import (
    FINGER_FLEXION_MAX,
    FINGER_PASSIVE_MAX,
    FINGER_PASSIVE_MIN,
    NOMINAL_HAND_MASS_KG,
    THUMB_ABDUCTION_MAX,
    THUMB_FLEXION_MAX,
    THUMB_PASSIVE_MAX,
    THUMB_PASSIVE_MIN,
)

ASSET_DIR = ROOT / "assets" / "xarm5_dh116"
SRC_DIR = ASSET_DIR / "src"

ADAPTER_R_BOTTOM = 0.0378   # xArm5 link5 flange face radius (measured on link5.stl)
ADAPTER_R_TOP = 0.0246      # DH116 base_link mounting ring radius (measured on base_link.STL)
ADAPTER_HEIGHT = 0.012
HAND_MOUNT_YAW = 0.0

# Leadshine 2026-04-09 ranges.  The nonlinear coupling is applied at runtime;
# URDF mimic is only linear and cannot represent the published quadratic term.
HAND_LIMITS = {
    "finger11": (0.0, THUMB_ABDUCTION_MAX),
    "finger12": (0.0, THUMB_FLEXION_MAX),
    "finger13": (THUMB_PASSIVE_MIN, THUMB_PASSIVE_MAX),
    "finger21": (0.0, FINGER_FLEXION_MAX),
    "finger22": (FINGER_PASSIVE_MIN, FINGER_PASSIVE_MAX),
    "finger31": (0.0, FINGER_FLEXION_MAX),
    "finger32": (FINGER_PASSIVE_MIN, FINGER_PASSIVE_MAX),
    "finger41": (0.0, FINGER_FLEXION_MAX),
    "finger42": (FINGER_PASSIVE_MIN, FINGER_PASSIVE_MAX),
    "finger51": (0.0, FINGER_FLEXION_MAX),
    "finger52": (FINGER_PASSIVE_MIN, FINGER_PASSIVE_MAX),
}
HAND_EFFORT, HAND_VELOCITY = 2.0, 3.0


def apply_current_hand_model(robot: ET.Element, prefix: str = "hand_") -> None:
    """Apply current limits and nominal palm mass to a generated URDF."""
    for joint in robot.findall("joint"):
        name = joint.get("name", "")
        base = name[len(prefix):] if name.startswith(prefix) else ""
        if base not in HAND_LIMITS:
            continue
        limit = joint.find("limit")
        lo, hi = HAND_LIMITS[base]
        limit.set("lower", f"{lo:.9g}")
        limit.set("upper", f"{hi:.9g}")
        limit.set("effort", f"{HAND_EFFORT}")
        limit.set("velocity", f"{HAND_VELOCITY}")
        # Native URDF mimic is linear.  Runtime code applies the complete
        # quadratic relationship, so a linear mimic constraint would conflict.
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)

    links = [link for link in robot.findall("link")
             if link.get("name", "").startswith(prefix)]
    base_link = next(link for link in links if link.get("name") == prefix + "base_link")
    total = sum(float(link.find("inertial/mass").get("value")) for link in links)
    if abs(total - NOMINAL_HAND_MASS_KG) < 1e-9:
        return
    base_inertial = base_link.find("inertial")
    base_mass = base_inertial.find("mass")
    old_base_mass = float(base_mass.get("value"))
    new_base_mass = old_base_mass + NOMINAL_HAND_MASS_KG - total
    if new_base_mass <= 0:
        raise ValueError(f"Cannot calibrate DH116 mass from {total} kg")
    base_mass.set("value", f"{new_base_mass:.9g}")
    inertia = base_inertial.find("inertia")
    scale = new_base_mass / old_base_mass
    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        inertia.set(key, f"{float(inertia.get(key)) * scale:.9g}")


def write_frustum_stl(path: Path, r0: float, r1: float, h: float, n: int = 96) -> None:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    b = np.stack([r0 * np.cos(ang), r0 * np.sin(ang), np.zeros(n)], 1)
    t = np.stack([r1 * np.cos(ang), r1 * np.sin(ang), np.full(n, h)], 1)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(b[i], b[j], t[j]), (b[i], t[j], t[i])]           # side, outward
        tris += [(b[j], b[i], [0, 0, 0]), (t[i], t[j], [0, 0, h])]  # caps
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for a, bb, c in tris:
            a, bb, c = map(np.asarray, (a, bb, c))
            nrm = np.cross(bb - a, c - a); nrm /= np.linalg.norm(nrm) + 1e-12
            f.write(struct.pack("<3f", *nrm) + struct.pack("<9f", *a, *bb, *c) + b"\0\0")


def _origin(xyz="0 0 0", rpy="0 0 0"):
    return ET.Element("origin", xyz=xyz, rpy=rpy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xarm-urdf", type=Path)
    ap.add_argument("--xarm-meshes", type=Path)
    ap.add_argument("--dh116-pkg", type=Path)
    ap.add_argument("--refresh-generated", action="store_true",
                    help="Apply the current hand model to the existing merged URDF.")
    a = ap.parse_args()

    if a.refresh_generated:
        out = ASSET_DIR / "xarm5_dh116.urdf"
        robot = ET.parse(out).getroot()
        apply_current_hand_model(robot)
        ET.indent(robot, space="  ")
        ET.ElementTree(robot).write(out, xml_declaration=True, encoding="utf-8")
        print(f"refreshed {out} with the Leadshine 2026-04-09 joint model")
        return
    if not all((a.xarm_urdf, a.xarm_meshes, a.dh116_pkg)):
        ap.error("normal builds require --xarm-urdf, --xarm-meshes and --dh116-pkg")

    # --- copy vendor sources -------------------------------------------------
    xarm_dst = SRC_DIR / "xarm5_description"
    hand_dst = SRC_DIR / "dh116_r_description"
    for d in (xarm_dst / "meshes", hand_dst / "meshes"):
        d.mkdir(parents=True, exist_ok=True)
    xarm = ET.parse(a.xarm_urdf).getroot()
    for m in xarm.iter("mesh"):
        rel = m.get("filename").replace("package://xarm_description/meshes/", "")
        dst = xarm_dst / "meshes" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.xarm_meshes / rel, dst)
        m.set("filename", f"src/xarm5_description/meshes/{rel}")
    shutil.copy2(a.xarm_urdf, xarm_dst / "xarm5_expanded.urdf")
    hand = ET.parse(next((a.dh116_pkg / "urdf").glob("*.urdf"))).getroot()
    for m in hand.iter("mesh"):
        name = m.get("filename").split("/")[-1]
        shutil.copy2(a.dh116_pkg / "meshes" / name, hand_dst / "meshes" / name)
        m.set("filename", f"src/dh116_r_description/meshes/{name}")
    shutil.copy2(next((a.dh116_pkg / "urdf").glob("*.urdf")), hand_dst / "DH116-R000-A1.urdf")
    adapter_stl = SRC_DIR / "flange_adapter.stl"
    write_frustum_stl(adapter_stl, ADAPTER_R_BOTTOM, ADAPTER_R_TOP, ADAPTER_HEIGHT)

    # --- merge ---------------------------------------------------------------
    robot = ET.Element("robot", name="xarm5_dh116")
    for mat in xarm.findall("material"):
        robot.append(copy.deepcopy(mat))
    adapter_mat = ET.SubElement(robot, "material", name="adapter_anodized")
    ET.SubElement(adapter_mat, "color", rgba="0.12 0.12 0.13 1")
    for el in xarm:
        if el.tag in ("gazebo", "transmission", "material"):
            continue
        if el.tag == "link" and el.get("name") == "world":
            continue
        if el.tag == "joint" and el.get("name") == "world_joint":
            continue
        robot.append(copy.deepcopy(el))

    # adapter link + joints
    link = ET.SubElement(robot, "link", name="flange_adapter")
    inertial = ET.SubElement(link, "inertial")
    inertial.append(_origin(xyz=f"0 0 {ADAPTER_HEIGHT/2}"))
    ET.SubElement(inertial, "mass", value="0.05")
    ET.SubElement(inertial, "inertia", ixx="1e-5", ixy="0", ixz="0", iyy="1e-5", iyz="0", izz="2e-5")
    for tag in ("visual", "collision"):
        v = ET.SubElement(link, tag)
        v.append(_origin())
        g = ET.SubElement(v, "geometry")
        ET.SubElement(g, "mesh", filename="src/flange_adapter.stl")
        if tag == "visual":
            ET.SubElement(v, "material", name="adapter_anodized")
    j = ET.SubElement(robot, "joint", name="joint_adapter", type="fixed")
    j.append(_origin()); ET.SubElement(j, "parent", link="link_eef"); ET.SubElement(j, "child", link="flange_adapter")
    j = ET.SubElement(robot, "joint", name="joint_hand_mount", type="fixed")
    j.append(_origin(xyz=f"0 0 {ADAPTER_HEIGHT}", rpy=f"0 0 {HAND_MOUNT_YAW}"))
    ET.SubElement(j, "parent", link="flange_adapter"); ET.SubElement(j, "child", link="hand_base_link")

    # hand, prefixed
    for el in hand:
        el = copy.deepcopy(el)
        if el.tag == "link":
            el.set("name", "hand_" + el.get("name"))
        elif el.tag == "joint":
            el.set("name", "hand_" + el.get("name"))
            el.find("parent").set("link", "hand_" + el.find("parent").get("link"))
            el.find("child").set("link", "hand_" + el.find("child").get("link"))
            lim = el.find("limit")
            base = el.get("name")[5:]
            if lim is not None and base in HAND_LIMITS:
                lo, hi = HAND_LIMITS[base]
                lim.set("lower", f"{lo}"); lim.set("upper", f"{hi}")
                lim.set("effort", f"{HAND_EFFORT}"); lim.set("velocity", f"{HAND_VELOCITY}")
        else:
            continue
        robot.append(el)

    apply_current_hand_model(robot)
    out = ASSET_DIR / "xarm5_dh116.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(out, xml_declaration=True, encoding="utf-8")
    n_links = len(robot.findall("link")); n_joints = len(robot.findall("joint"))
    print(f"wrote {out} ({n_links} links, {n_joints} joints); adapter {adapter_stl.name}")


if __name__ == "__main__":
    main()
