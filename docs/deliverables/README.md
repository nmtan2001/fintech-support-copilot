# Engineering assessment documents

engineering-report.pdf is a seven-page validation report.
engineering-presentation.pptx contains ten editable slides.
Original submission/ documents were preserved. These documents describe the three verified application deployments, measured
model quality, limitations and reproducible release procedure. See the evidence
index for exact commits, digests, workflow URLs and runtime credential expiry.

The report source is report-source.json. Install reportlab==5.1.0 in a separate
document environment and run build_report.py to regenerate the PDF.

The slide source is slides-source.json. build_slides.cjs uses PptxGenJS,
Playwright 1.57.0, Sharp and the presentation skill's html2pptx helper. Supply the
helper module and a temporary preview directory as its two arguments. The helper
validates HTML bounds; screenshots of all ten slides were visually inspected.
The compiled PowerPoint is usable without that authoring environment.

The application is local, with no public endpoint. The teacher still requires
a fresh source change during assessment; follow ../assessment-rehearsal.md.
