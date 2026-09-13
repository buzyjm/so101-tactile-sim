"""Cut the hand out of the merged xArm5+DH116 URDF as a stand-alone robot.

    python scripts/dh116_hand/build_hand_urdf.py

Keeps every ``hand_*`` link and ``hand_finger*`` joint (with the April 2026
vendor limits, nominal mass, and effort/velocity fixes already applied), roots the
robot at ``hand_base_link`` and points the meshes back at the vendor STLs
under assets/xarm5_dh116/src.  Output: assets/dh116_hand/dh116_hand.urdf.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "assets" / "xarm5_dh116" / "xarm5_dh116.urdf"
OUT_DIR = ROOT / "assets" / "dh116_hand"
OUT = OUT_DIR / "dh116_hand.urdf"


def main() -> None:
    src = ET.parse(MERGED).getroot()
    robot = ET.Element("robot", name="dh116_hand")
    for material in src.findall("material"):
        robot.append(material)
    kept_links = 0
    for link in src.findall("link"):
        if not link.get("name", "").startswith("hand_"):
            continue
        for mesh in link.iter("mesh"):
            name = Path(mesh.get("filename")).name
            mesh.set("filename", f"../xarm5_dh116/src/dh116_r_description/meshes/{name}")
        robot.append(link)
        kept_links += 1
    kept_joints = 0
    for joint in src.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if not (parent.startswith("hand_") and child.startswith("hand_")):
            continue
        robot.append(joint)
        kept_joints += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(OUT, encoding="unicode", xml_declaration=True)
    print(f"{OUT}: {kept_links} links, {kept_joints} joints")


if __name__ == "__main__":
    main()
