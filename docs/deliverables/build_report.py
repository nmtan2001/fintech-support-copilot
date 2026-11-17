"""Render the reviewable engineering report source to a seven-page PDF."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parent
DARK = colors.HexColor("#19352B")
ACCENT = colors.HexColor("#D9EAE1")


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
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(DARK)
        canvas.drawString(43, 28, "Fintech Support Copilot | Engineering validation | 17 November 2026")
        canvas.drawRightString(550, 28, str(doc.page))
    document = SimpleDocTemplate(str(ROOT / "engineering-report.pdf"),
                                 leftMargin=43, rightMargin=43, topMargin=38, bottomMargin=45,
                                 title=data["title"], author="Fintech Support Copilot")
    document.build(flow, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
