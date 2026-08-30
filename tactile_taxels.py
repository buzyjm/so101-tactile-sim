"""Load the DP-S2015-Elite 52-taxel geometry from its source spreadsheet."""

from __future__ import annotations

from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COORDINATE_PATH = PROJECT_ROOT / "assets" / "tactile" / "array.xlsx"
TAXEL_COUNT = 52
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def load_taxel_positions_mm(
    path: str | Path = DEFAULT_COORDINATE_PATH,
) -> tuple[tuple[float, float, float], ...]:
    """Return taxels 1..52 as XYZ positions in the native CAD frame (mm)."""
    coordinate_path = Path(path)
    with ZipFile(coordinate_path) as workbook:
        sheet = ElementTree.fromstring(
            workbook.read("xl/worksheets/sheet1.xml")
        )

    rows = []
    namespace = {"x": _MAIN_NS}
    for row in sheet.findall(".//x:sheetData/x:row", namespace):
        values: dict[int, float] = {}
        for cell in row.findall("x:c", namespace):
            value = cell.find("x:v", namespace)
            if value is None or value.text is None:
                continue
            values[_column_index(cell.attrib["r"])] = float(value.text)
        if all(index in values for index in range(4)):
            rows.append(
                (
                    int(values[0]),
                    (values[1], values[2], values[3]),
                )
            )

    expected_ids = list(range(1, TAXEL_COUNT + 1))
    actual_ids = [taxel_id for taxel_id, _ in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Expected taxel IDs 1..{TAXEL_COUNT}, got {actual_ids} in "
            f"{coordinate_path}"
        )
    return tuple(position for _, position in rows)


def load_taxel_positions_m(
    path: str | Path = DEFAULT_COORDINATE_PATH,
) -> tuple[tuple[float, float, float], ...]:
    """Return taxels 1..52 as XYZ positions in the native CAD frame (m)."""
    return tuple(
        tuple(coordinate / 1000.0 for coordinate in position)
        for position in load_taxel_positions_mm(path)
    )
