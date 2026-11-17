# Engineering assessment documents

engineering-report.pdf is a seven-page validation report.
engineering-presentation.pptx contains ten editable slides.
Original submission/ documents were preserved. These new documents distinguish
verified measurements from pending final releases. Refresh the status/evidence
paragraphs after later deployment runs rather than claiming unverified results.

The report source is report-source.json. Install reportlab==5.1.0 in a separate
document environment and run build_report.py to regenerate the PDF.

The slide source is slides-source.json. build_slides.cjs uses PptxGenJS,
Playwright 1.57.0, Sharp and the presentation skill's html2pptx helper. Supply the
helper module and a temporary preview directory as its two arguments. The helper
validates HTML bounds; screenshots of all ten slides were visually inspected.
The compiled PowerPoint is usable without that authoring environment.

The slide deck intentionally does not claim an installed public endpoint or
completed three-version release sequence until actual evidence is added.
