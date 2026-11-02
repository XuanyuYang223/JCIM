"""Draw the frozen molecular motifs, combination classes, and label budgets."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
SVG_NS = "http://www.w3.org/2000/svg"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motif_image(smiles, vector=False):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule.GetNumHeavyAtoms() == 9
    assert sum(atom.GetSymbol() == "C" for atom in molecule.GetAtoms()) == 7
    assert sum(atom.GetSymbol() == "O" for atom in molecule.GetAtoms()) == 2
    # A dummy atom depicts the ring-side attachment; it is not a tenth heavy atom.
    editable = Chem.RWMol(molecule)
    attachment = editable.AddAtom(Chem.Atom(0))
    editable.AddBond(0, attachment, Chem.BondType.SINGLE)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    drawer = (rdMolDraw2D.MolDraw2DSVG if vector else rdMolDraw2D.MolDraw2DCairo)(800, 360)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.atomLabels[attachment] = "*"
    options.padding = .15
    drawer.DrawMolecule(molecule, highlightAtoms=[0], highlightAtomColors={0: (.58, .78, .91)})
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main():
    source = ROOT / "experiments/hard_v2/molecules.jsonl"
    source_hash = digest(source)
    examples = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        examples.setdefault(record["y_family"], record["motif_smiles"])
    assert set(examples) == set(range(8))
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                         "svg.hashsalt": "jcim-controlled-design-v1"})
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    figure.text(.035, .955, "A   Eight known connectivity motifs", fontsize=16, weight="bold")
    figure.text(.035, .915, "Each: 7 carbon + 2 oxygen atoms\nBlue atom: attachment root; *: ring-side connection", fontsize=11)
    positions, vectors = {}, {}
    for family in range(8):
        row, column = divmod(family, 2)
        axis = figure.add_axes([.025 + column * .205, .69 - row * .18, .205, .16])
        image = axis.imshow(Image.open(BytesIO(motif_image(examples[family]))))
        image.set_gid(f"motif-vector-{family}")
        axis.text(.5, .01, f"M{family}", transform=axis.transAxes, ha="center", va="top", fontsize=12)
        axis.axis("off")
        positions[family] = axis
        vectors[family] = motif_image(examples[family], vector=True)

    matrix_axis = figure.add_axes([.465, .25, .245, .59])
    colors = ("#c7dfea", "#f2ddc4", "#ddd5eb")
    cell_labels = ("A", "B", "T")
    matrix = []
    for motif in range(8):
        current = []
        for position in range(3):
            cell = (motif + position) % 3
            current.append(cell)
            matrix_axis.add_patch(Rectangle((position, motif), 1, 1, facecolor=colors[cell], edgecolor="white", linewidth=2))
            matrix_axis.text(position + .5, motif + .5, cell_labels[cell], ha="center", va="center", fontsize=15, weight="bold")
        matrix.append(current)
    matrix_axis.set(xlim=(0, 3), ylim=(8, 0), xticks=[.5, 1.5, 2.5], xticklabels=["ortho", "meta", "para"],
                    yticks=[index + .5 for index in range(8)], yticklabels=[f"M{index}" for index in range(8)])
    matrix_axis.tick_params(length=0, pad=8)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)
    figure.text(.46, .955, "B   Combination classes", fontsize=16, weight="bold")
    figure.text(.46, .915, "c = (motif + position) mod 3\nEach motif has one source position per experience", fontsize=11)
    figure.text(.46, .19, "A: source experience A\nB: source experience B\nT: common unseen-combination class", fontsize=11, linespacing=1.6, va="top")
    figure.text(.46, .07, "Canonical molecular identity is split separately.\nCells describe combinations, not repeated molecules.", fontsize=10, color="#444444")

    figure.text(.755, .955, "C   Distinct supervision", fontsize=16, weight="bold")
    cards = [
        (.68, "Original-source prediction", "1,386 original training identities\nNo new-combination labels\nOracle boundaries only when specified"),
        (.43, "Directed structural probe", "256 balanced labels at one position\nTest the other positions\nExtra diagnostic supervision"),
        (.18, "Labeled graph adaptation", "32 new + 32 original identities\nMatched trained / random features\nNew-combination labels are supplied"),
    ]
    for y, title, body in cards:
        box = FancyBboxPatch((.75, y), .225, .19, boxstyle="round,pad=.012", transform=figure.transFigure,
                             facecolor="#f5f7f8", edgecolor="#a6afb5", linewidth=1.1)
        figure.patches.append(box)
        figure.text(.765, y + .145, title, fontsize=12, weight="bold")
        figure.text(.765, y + .10, body, fontsize=10.3, va="top", linespacing=1.6)
    figure.text(.755, .065, "Readability, adaptation, and source-only\nprediction answer different questions.", fontsize=11, weight="bold")

    figure.canvas.draw()
    png = OUT / "controlled_design.png"
    svg = OUT / "controlled_design.svg"
    figure.savefig(png, dpi=300)
    figure.savefig(svg, metadata={"Date": None})
    # Replace the temporary raster chemical drawings with native RDKit vectors.
    tree = ET.parse(svg)
    xml = tree.getroot()
    width, height = figure.get_size_inches() * 72
    for family, axis in positions.items():
        raster = next(node for node in xml.iter() if node.get("id") == f"motif-vector-{family}")
        parent = next(node for node in xml.iter() if raster in list(node))
        position = list(parent).index(raster)
        parent.remove(raster)
        group = ET.Element(f"{{{SVG_NS}}}g", {"id": f"motif-vector-{family}"})
        parent.insert(position, group)
        box = axis.get_position()
        nested = ET.SubElement(group, f"{{{SVG_NS}}}svg", {
            "x": str(box.x0 * width), "y": str((1 - box.y1) * height),
            "width": str(box.width * width), "height": str(box.height * height), "viewBox": "0 0 800 360"})
        drawing = ET.fromstring(vectors[family])
        for child in list(drawing):
            nested.append(child)
    assert not any(node.tag.endswith("}image") for node in xml.iter())
    assert sum(node.tag.endswith("}svg") for node in xml.iter()) == 9
    ET.register_namespace("", SVG_NS)
    tree.write(svg, encoding="utf-8", xml_declaration=True)
    plt.close(figure)
    assert digest(source) == source_hash
    counts = {str(cell): sum(value == cell for row in matrix for value in row) for cell in range(3)}
    assert set(counts.values()) == {8}
    metadata = {"source": source.relative_to(ROOT).as_posix(), "source_sha256": source_hash,
                "motifs": examples, "combination_classes": matrix, "class_cell_counts": counts,
                "root_annotation": "First motif atom highlighted; dummy star denotes ring attachment and is not a heavy atom.",
                "svg_chemical_structure_format": "Native RDKit vector paths, not embedded raster images.",
                "generator_sha256": digest(__file__), "outputs": {path.name: digest(path) for path in (png, svg)}}
    (OUT / "CONTROLLED_DESIGN_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Built controlled-design PNG/SVG from frozen motif identities; source bytes unchanged.")


if __name__ == "__main__":
    main()
