"""Export manuscript diagrams as publication-quality vector PDFs.

All text and drawing primitives are written directly to PDF with ReportLab, so the
figures remain sharp at arbitrary zoom.  Run from the repository root with:

    python tools/export_vector_figures.py
"""

from pathlib import Path
from textwrap import wrap

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black, white


OUT = Path(__file__).resolve().parents[1] / "artifacts/diagrams"
PAGE = (720, 360)

NAVY = HexColor("#17324D")
BLUE = HexColor("#DDEEFF")
CYAN = HexColor("#DDF5F2")
GREEN = HexColor("#E0F2DF")
GOLD = HexColor("#FFF0C2")
ORANGE = HexColor("#FFE3CB")
RED = HexColor("#F8D7DA")
GRAY = HexColor("#EEF1F4")
DARK_GRAY = HexColor("#4A5560")


def new_pdf(name, size=PAGE):
    c = canvas.Canvas(str(OUT / name), pagesize=size, pageCompression=1)
    c.setTitle(name.replace("_", " ").replace(".pdf", ""))
    return c


def finish(c):
    c.showPage()
    c.save()


def line_wrap(text, max_width, font="Helvetica", size=12):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def text_block(
    c,
    text,
    x,
    y,
    width,
    size=12,
    leading=None,
    font="Helvetica",
    color=black,
    align="center",
    max_lines=None,
):
    leading = leading or size * 1.25
    lines = line_wrap(text, width, font, size)
    if max_lines:
        lines = lines[:max_lines]
    c.setFillColor(color)
    c.setFont(font, size)
    for i, line in enumerate(lines):
        yy = y - i * leading
        if align == "center":
            c.drawCentredString(x + width / 2, yy, line)
        elif align == "right":
            c.drawRightString(x + width, yy, line)
        else:
            c.drawString(x, yy, line)
    return len(lines) * leading


def box(
    c,
    x,
    y,
    w,
    h,
    fill,
    title,
    body="",
    title_size=14,
    body_size=11,
    border=NAVY,
    radius=8,
):
    c.setFillColor(fill)
    c.setStrokeColor(border)
    c.setLineWidth(1.3)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
    text_block(
        c,
        title,
        x + 8,
        y + h - 22,
        w - 16,
        title_size,
        font="Helvetica-Bold",
        color=NAVY,
    )
    if body:
        text_block(
            c,
            body,
            x + 10,
            y + h - 52,
            w - 20,
            body_size,
            leading=body_size * 1.28,
            color=DARK_GRAY,
        )


def arrow(c, x1, y1, x2, y2, color=NAVY, dashed=False, width=1.8):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(5, 4)
    c.line(x1, y1, x2, y2)
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    ah = 8
    for delta in (2.65, -2.65):
        c.line(x2, y2, x2 + ah * math.cos(ang + delta), y2 + ah * math.sin(ang + delta))
    c.restoreState()


def header(c, title, subtitle=None, width=720, height=360):
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(24, height - 30, title)
    if subtitle:
        c.setFillColor(DARK_GRAY)
        c.setFont("Helvetica", 11)
        c.drawString(24, height - 48, subtitle)


def figure_1():
    c = new_pdf("Figure_1.pdf")
    header(
        c,
        "Architectural paradigms for multimodal efficiency",
        "Compute savings depend on where sparsity is introduced and what must still move through memory.",
    )
    xs = [24, 256, 488]
    fills = [BLUE, GREEN, GOLD]
    titles = ["A  Dense baseline", "B  Server-side MoE", "C  Edge-oriented composition"]
    bodies = [
        "All visual tokens traverse dense layers. Cost grows with image resolution, video length, and context size.",
        "A router activates a subset of experts. Arithmetic is sparse, but the full expert bank still incurs storage and movement costs.",
        "Token compression, sparse routing, and low-precision experts are combined. Benefit must be measured end to end on the target device.",
    ]
    for x, fill, title, body in zip(xs, fills, titles, bodies):
        box(c, x, 62, 208, 226, fill, title, body, title_size=15, body_size=12)
    for x in (232, 464):
        arrow(c, x - 10, 175, x + 10, 175, color=DARK_GRAY)
    finish(c)


def figure_2():
    c = new_pdf("Figure_2.pdf", (720, 500))
    header(c, "Taxonomy of efficient multimodal intelligence", width=720, height=500)
    rows = [
        (
            "Vision encoder sparsification",
            "Merging; self-supervised redundancy",
            "ToMe, ATM, AdaMerge, ToSA, MAE, VideoMAE",
            BLUE,
        ),
        (
            "MLLM sequence compression",
            "Multi-stage dropping; learned selection",
            "MustDrop, HiRED, ZipVL, GreedyPrune, EvoComp",
            ORANGE,
        ),
        (
            "KV-cache optimisation",
            "Dependency-aware eviction; semantic memory",
            "GraphKV, MadaKV, H2O, SnapKV, RetentiveKV",
            GOLD,
        ),
        (
            "MoE routing dynamics",
            "Load balance; stable and temporal routing",
            "ReMoE, auxiliary-free routing, eigen-router, TRC",
            GREEN,
        ),
        (
            "Edge quantization and systems",
            "Routing-aware PTQ; expert storage",
            "EAQuant, VSRAQ, EdgeMoE, ZipMoE, Janus",
            GRAY,
        ),
    ]
    y = 354
    for title, mechanisms, methods, fill in rows:
        box(c, 24, y, 205, 64, fill, title, title_size=13)
        box(
            c,
            256,
            y,
            205,
            64,
            white,
            mechanisms,
            title_size=12,
            border=HexColor("#8492A0"),
        )
        box(
            c,
            488,
            y,
            208,
            64,
            white,
            methods,
            title_size=11,
            border=HexColor("#8492A0"),
        )
        arrow(c, 231, y + 32, 252, y + 32, color=DARK_GRAY, width=1.3)
        arrow(c, 463, y + 32, 484, y + 32, color=DARK_GRAY, width=1.3)
        y -= 74
    c.setFillColor(DARK_GRAY)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(126, 438, "Operational bottleneck")
    c.drawCentredString(358, 438, "Representative mechanism")
    c.drawCentredString(592, 438, "Representative methods")
    finish(c)


def figure_3():
    c = new_pdf("Figure_3.pdf")
    header(
        c,
        "Two visual-token reduction mechanisms",
        "The diagrams describe different information-preservation behaviours, not a universal ranking.",
    )
    box(
        c,
        24,
        66,
        320,
        222,
        BLUE,
        "A  Token dropping",
        "Selected tokens are removed permanently. Remaining tokens keep their identity, but deleted spatial or semantic evidence cannot be recovered.",
        title_size=16,
        body_size=12,
    )
    box(
        c,
        376,
        66,
        320,
        222,
        GREEN,
        "B  Token merging",
        "Similar token pairs are replaced by aggregate representations. Token count falls while some source information is combined rather than discarded.",
        title_size=16,
        body_size=12,
    )
    for base_x, keep, drop in [(62, BLUE, RED), (414, GREEN, GREEN)]:
        for row in range(2):
            for col in range(5):
                x = base_x + col * 43
                y = 92 + row * 34
                c.setFillColor(drop if (row + col) % 4 == 0 else keep)
                c.setStrokeColor(NAVY)
                c.rect(x, y, 26, 22, stroke=1, fill=1)
    c.setFillColor(DARK_GRAY)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(184, 52, "Removal preserves fewer token identities")
    c.drawCentredString(536, 52, "Aggregation preserves combined representations")
    finish(c)


def figure_5():
    c = new_pdf("Figure_5.pdf")
    header(
        c,
        "Quantization failure regimes in VLM and MoE inference",
        "Evidence strength and mitigation differ across gradual, structural, and catastrophic failures.",
    )
    data = [
        (
            "Signal degradation",
            "Gradual loss of fidelity",
            "Evidence: 4-bit evaluated regimes",
            "Mitigation: calibration or fine-tuning",
            GOLD,
        ),
        (
            "Routing inconsistency",
            "Expert selection changes",
            "Evidence: routing-aware PTQ recovery",
            "Mitigation: preserve router structure",
            ORANGE,
        ),
        (
            "Computation collapse",
            "Numerical instability",
            "Evidence: 2-bit evaluated regimes",
            "Mitigation: mixed precision or repair",
            RED,
        ),
    ]
    for i, (title, failure, evidence, mitigation, fill) in enumerate(data):
        x = 24 + i * 232
        box(
            c,
            x,
            68,
            208,
            220,
            fill,
            title,
            f"What breaks: {failure}. {evidence}. {mitigation}.",
            title_size=14,
            body_size=12,
        )
    finish(c)


def figure_6():
    c = new_pdf("Figure_6.pdf")
    header(
        c,
        "KV-cache growth under a fixed edge-memory budget",
        "The break point is device- and implementation-specific; growth is linear in retained tokens for a fixed model.",
    )
    left, bottom, right, top = 82, 62, 680, 278
    c.setStrokeColor(NAVY)
    c.setLineWidth(1.5)
    c.line(left, bottom, left, top)
    c.line(left, bottom, right, bottom)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(NAVY)
    c.drawCentredString((left + right) / 2, 34, "Retained multimodal tokens")
    c.saveState()
    c.translate(25, (bottom + top) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "KV-cache memory")
    c.restoreState()
    budget_y = 226
    c.setStrokeColor(HexColor("#B22222"))
    c.setDash(6, 4)
    c.line(left, budget_y, right, budget_y)
    c.setDash()
    c.setFillColor(HexColor("#B22222"))
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(right, budget_y + 7, "Device memory budget")
    c.setStrokeColor(HexColor("#2471A3"))
    c.setLineWidth(4)
    c.line(left + 12, bottom + 15, right - 24, top - 14)
    c.setFillColor(HexColor("#2471A3"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(405, 187, "Dense cache growth")
    cross_x = (
        left
        + 12
        + (budget_y - (bottom + 15))
        * ((right - 24) - (left + 12))
        / ((top - 14) - (bottom + 15))
    )
    c.setFillColor(HexColor("#B22222"))
    c.circle(cross_x, budget_y, 6, stroke=0, fill=1)
    c.setFont("Helvetica", 11)
    c.drawRightString(
        right - 8, budget_y - 18, "Action: offload / recompute / compress / evict"
    )
    finish(c)


def figure_7():
    c = new_pdf("Figure_7.pdf", (720, 420))
    header(c, "Proposed cross-stage interaction framework", width=720, height=420)
    stages = [
        ("Compression", "retained tokens", BLUE),
        ("MoE routing", "expert assignments", GREEN),
        ("Quantized inference", "precision and bytes", GOLD),
        ("Hardware execution", "latency and energy", GRAY),
    ]
    xs = [20, 200, 380, 560]
    for x, (title, body, fill) in zip(xs, stages):
        box(c, x, 236, 140, 92, fill, title, body, title_size=14, body_size=11)
    labels = [
        ("Interface B", "representation shift", True),
        ("Interface C", "routing / precision", False),
        ("Interface D", "model / hardware", False),
    ]
    for i, (label, note, dashed) in enumerate(labels):
        x1, x2 = xs[i] + 140, xs[i + 1]
        arrow(c, x1 + 4, 282, x2 - 4, 282, color=DARK_GRAY, dashed=dashed)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString((x1 + x2) / 2, 350, label)
        c.setFont("Helvetica", 9)
        c.drawCentredString((x1 + x2) / 2, 337, note)
    box(
        c,
        78,
        94,
        230,
        82,
        CYAN,
        "Interface A",
        "Visual input to compression",
        title_size=14,
        body_size=11,
    )
    box(
        c,
        412,
        94,
        230,
        82,
        ORANGE,
        "KV cache: persistent state",
        "Stores earlier states; retention and size feed hardware pressure",
        title_size=13,
        body_size=10,
    )
    arrow(c, 150, 176, 100, 232, color=DARK_GRAY)
    arrow(c, 527, 176, 621, 232, color=DARK_GRAY, dashed=True)
    c.setStrokeColor(HexColor("#B22222"))
    c.setDash(6, 4)
    c.setLineWidth(1.5)
    c.line(621, 88, 621, 58)
    c.line(621, 58, 99, 58)
    c.line(99, 58, 99, 88)
    c.setDash()
    c.setFillColor(HexColor("#B22222"))
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(360, 42, "Hardware budget feeds back on all upstream choices")
    c.setFillColor(DARK_GRAY)
    c.setFont("Helvetica", 9)
    c.drawString(
        24,
        12,
        "Solid arrows: directly evidenced stage connection.  Dashed arrows: inferred or proposed interaction requiring joint evaluation.",
    )
    finish(c)


def edge_pareto():
    """Re-export the existing illustrative Pareto plot with readable, separated labels."""
    width, height = 744, 451
    c = canvas.Canvas(
        str(OUT / "Figure_4.pdf"), pagesize=(width, height), pageCompression=1
    )
    c.setTitle("Illustrative multimodal edge-efficiency Pareto view")
    left, right, top, bottom = 54, 570, 56, 408

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, 430, "Multimodal Edge Efficiency Pareto View")
    c.setFont("Helvetica", 11)
    c.setFillColor(DARK_GRAY)
    c.drawCentredString(
        width / 2, 413, "OpenVLM MMBench V1.1 score versus illustrative edge throughput"
    )

    # Axes and grid.  Coordinates reproduce the original illustrative data placement.
    x_ticks = [(111, "10"), (208, "15"), (306, "20"), (403, "25"), (501, "30")]
    y_ticks = [
        (60, "85"),
        (111, "80"),
        (163, "75"),
        (214, "70"),
        (266, "65"),
        (317, "60"),
        (369, "55"),
    ]
    c.setStrokeColor(HexColor("#D3D8DE"))
    c.setLineWidth(0.7)
    c.setDash(2, 2)
    for x, _ in x_ticks:
        c.line(x, height - bottom, x, height - top)
    for y, _ in y_ticks:
        c.line(left, height - y, right, height - y)
    c.setDash()
    c.setStrokeColor(NAVY)
    c.setLineWidth(1.2)
    c.rect(left, height - bottom, right - left, bottom - top, stroke=1, fill=0)
    c.setFillColor(black)
    c.setFont("Helvetica", 9)
    for x, label in x_ticks:
        c.drawCentredString(x, height - bottom - 15, label)
    for y, label in y_ticks:
        c.drawRightString(left - 7, height - y - 3, label)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(
        (left + right) / 2,
        16,
        "Illustrative edge throughput (tokens/s)  -  higher is better",
    )
    c.saveState()
    c.translate(15, (height - top + height - bottom) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "MMBench V1.1 score  -  higher is better")
    c.restoreState()

    c.setStrokeColor(HexColor("#9AA1A8"))
    c.setLineWidth(1.1)
    c.setDash(5, 4)
    c.line(306, height - bottom, 306, height - top)
    c.setDash()
    c.setFillColor(HexColor("#737A82"))
    c.setFont("Helvetica", 9)
    c.drawString(315, height - 150, "Illustrative real-time")
    c.drawString(315, height - 162, "throughput threshold")

    dense = HexColor("#D6272D")
    compact = HexColor("#FF7F0E")
    sparse = HexColor("#1F77B4")
    points = [
        # label, centre x, centre y in top-origin coordinates, radius, colour, label x, label y
        ("LLaVA-1.5", 72, 385, 9, dense, 82, 373),
        ("Ola-7B", 74, 65, 10, dense, 56, 34),
        ("Ovis2-8B", 78, 73, 10, dense, 105, 50),
        ("InternVL3-8B", 84, 88, 9, dense, 45, 118),
        ("LLaVA-OV", 88, 101, 9, dense, 55, 162),
        ("Qwen2.5-7B", 96, 87, 9, dense, 88, 145),
        ("MiniCPM-o", 127, 104, 9, dense, 155, 131),
        ("R-4B", 213, 82, 7, compact, 255, 54),
        ("Qwen2.5-3B", 219, 143, 6, compact, 185, 166),
        ("Ovis-U1", 248, 131, 6, compact, 254, 153),
        ("InternVL3-2B", 287, 130, 5, compact, 293, 119),
        ("InternVL3-1B", 443, 231, 4, compact, 449, 221),
        ("SmolVLM2", 521, 303, 5, compact, 476, 330),
        ("DS-VL2-Tiny", 345, 203, 6, sparse, 352, 226),
        ("DS-VL2-Small", 209, 111, 13, sparse, 229, 109),
        ("DS-VL2", 141, 98, 16, sparse, 178, 55),
    ]
    c.setLineWidth(0.7)
    for label, px, py, radius, color, lx, ly in points:
        cy, label_y = height - py, height - ly
        c.setStrokeColor(HexColor("#FFFFFF"))
        c.setFillColor(color)
        c.circle(px, cy, radius, stroke=1, fill=1)
        c.setStrokeColor(HexColor("#7A838C"))
        c.line(px, cy, lx, label_y - 2)
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 9.2)
        c.drawString(lx, label_y, label)

    # Compact legend in the reserved right margin.
    c.setFillColor(white)
    c.setStrokeColor(HexColor("#B8C0C8"))
    c.roundRect(588, 175, 145, 92, 5, stroke=1, fill=1)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(600, 248, "Architecture type")
    for y, color, label in [
        (228, dense, "Dense (FP16)"),
        (207, compact, "Compact dense (FP16)"),
        (186, sparse, "Sparse MoE (FP16)"),
    ]:
        c.setFillColor(color)
        c.rect(600, y, 20, 7, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawString(628, y - 1, label)

    c.setFillColor(DARK_GRAY)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(330, 52, "Bubble area represents approximate FP16 model footprint.")
    finish(c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    figure_5()
    figure_6()
    figure_7()
    edge_pareto()
    for path in sorted(OUT.glob("Figure_*.pdf")):
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")
    edge = OUT / "Figure_4.pdf"
    print(f"wrote {edge.name} ({edge.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
