"""Render the reviewable engineering report source to a PDF."""
import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parent
DARK = colors.HexColor("#19352B")
ACCENT = colors.HexColor("#D9EAE1")



def runtime_diagram():
    diagram = Drawing(450, 165)
    def box(x, y, width, text):
        diagram.add(Rect(x, y, width, 28, fillColor=ACCENT, strokeColor=DARK, strokeWidth=0.7))
        diagram.add(String(x + width / 2, y + 10, text, textAnchor="middle", fontName="Helvetica", fontSize=9, fillColor=DARK))
    def arrow(x1, y1, x2, y2):
        diagram.add(Line(x1, y1, x2, y2, strokeColor=DARK, strokeWidth=0.8))
        if y2 < y1:
            diagram.add(Polygon([x2, y2, x2 - 3, y2 + 6, x2 + 3, y2 + 6], fillColor=DARK, strokeColor=DARK))
        else:
            diagram.add(Polygon([x2, y2, x2 - 3, y2 - 6, x2 + 3, y2 - 6], fillColor=DARK, strokeColor=DARK))
    diagram.add(String(0, 153, "Kubernetes runtime: two CPU replicas, synthetic records only", fontName="Helvetica-Bold", fontSize=9, fillColor=DARK))
    box(175, 115, 100, "ClusterIP Service")
    box(70, 60, 130, "API / model replica A")
    box(280, 60, 130, "API / model replica B")
    box(40, 0, 130, "ConfigMap + Secrets")
    box(280, 0, 130, "Pinned private model")
    arrow(210, 115, 135, 88)
    arrow(240, 115, 345, 88)
    arrow(100, 28, 115, 60)
    arrow(140, 28, 310, 60)
    arrow(310, 28, 180, 60)
    arrow(345, 28, 365, 60)
    return diagram


def main():
    data = json.loads((ROOT / "report-source.json").read_text())
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=11, leading=17,
                          textColor=colors.HexColor("#263C35"), spaceAfter=16, alignment=TA_LEFT)
    title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=23,
                           leading=29, textColor=DARK, spaceAfter=20)
    label = ParagraphStyle("label", parent=body, fontSize=9, leading=12)
    flow = []
    for index, section in enumerate(data["sections"]):
        if index:
            flow.append(PageBreak())
        flow.append(Paragraph(escape(data["title"].upper()), label))
        flow.append(Paragraph(escape(section["title"]), title))
        flow.append(Paragraph(escape(data["status"]), label))
        flow.append(Spacer(1, 10))
        for text in section["paragraphs"]:
            flow.append(Paragraph(escape(text), body))
        if index == 1:
            rows = [["Source + tests", "Image + model gate", "Registry digest", "Rollout + verify"]]
            table = Table(rows, colWidths=[1.55 * inch] * 4)
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT),
                                       ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
                                       ("FONTSIZE", (0, 0), (-1, -1), 9),
                                       ("TOPPADDING", (0, 0), (-1, -1), 14),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
            flow.append(table)
            flow.append(Spacer(1, 14))
            flow.append(runtime_diagram())
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(DARK)
        canvas.drawString(43, 28, "Fintech Support Copilot | Engineering validation | " + date.today().strftime("%d %B %Y"))
        canvas.drawRightString(550, 28, str(doc.page))
    document = SimpleDocTemplate(str(ROOT / "engineering-report.pdf"),
                                 leftMargin=43, rightMargin=43, topMargin=38, bottomMargin=45,
                                 title=data["title"], author="Fintech Support Copilot")
    document.build(flow, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
