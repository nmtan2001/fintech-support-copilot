/* Use the html2pptx helper supplied by the presentation skill. */
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { chromium } = require('playwright');
const helper = process.argv[2];
const preview = process.argv[3];
if (!helper || !preview) throw new Error('Usage: node build_slides.cjs HTML2PPTX_MODULE PREVIEW_DIRECTORY');
const html2pptx = require(path.resolve(helper));
const escape = (s) => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const root = __dirname;
const source = JSON.parse(fs.readFileSync(path.join(root, 'slides-source.json'), 'utf8'));
fs.mkdirSync(preview, { recursive: true });
(async () => {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.title = 'Fintech Support Copilot — DevOps and MLOps';
  pptx.subject = 'Engineering validation and practical release evidence';
  pptx.author = 'Fintech Support Copilot';
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
  for (const [index, item] of source.entries()) {
    const card = (title, entries) => `<div class="card"><h2>${escape(title)}</h2><ul>${entries.map(s => `<li>${escape(s)}</li>`).join('')}</ul></div>`;
    const html = `<!doctype html><html><head><meta charset="utf-8"><style>
      *{box-sizing:border-box} html{background:#132820}body{width:720pt;height:405pt;margin:0;padding:28pt 32pt;display:flex;flex-direction:column;background:#132820;color:#F5F8F5;font-family:Arial}
      p{margin:0} .kicker{font-family:'Courier New';font-size:10pt;letter-spacing:1pt;color:#B5E3CE;margin-bottom:12pt}
      h1{font-size:29pt;line-height:1.12;margin:0 0 23pt;max-width:650pt} .cards{display:flex;gap:18pt;flex:1}
      .card{width:319pt;background:#214035;padding:20pt 18pt;border-top:3pt solid #B5E3CE}
      h2{font-size:21pt;line-height:1.1;color:#ECC08E;margin:0 0 19pt}ul{margin:0;padding-left:17pt}li{font-size:16pt;line-height:1.3;margin-bottom:13pt}
      .footer{font-size:9pt;color:#B4C8BE;margin-top:15pt}
    </style></head><body><p class="kicker">${escape(item.kicker)}</p><h1>${escape(item.title)}</h1><div class="cards">${card(item.leftTitle,item.left)}${card(item.rightTitle,item.right)}</div><p class="footer">Engineering validation snapshot | Synthetic data | ${String(index + 1).padStart(2,'0')} / 10</p></body></html>`;
    const htmlFile = path.join(preview, `slide-${index + 1}.html`);
    fs.writeFileSync(htmlFile, html);
    const { slide } = await html2pptx(htmlFile, pptx, { tmpDir: preview });
    slide.addNotes('Explain the distinction between implemented controls and verified release evidence. See docs/operations.md and docs/assignment-evidence.md for exact commands and run links.');
    await page.goto('file://' + htmlFile);
    await page.screenshot({ path: path.join(preview, `slide-${index + 1}.png`) });
  }
  await browser.close();
  await pptx.writeFile({ fileName: path.join(root, 'engineering-presentation.pptx') });
  console.log('Created 10 slides with HTML overflow validation and PNG previews.');
})().catch(error => { console.error(error); process.exit(1); });
