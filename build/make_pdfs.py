"""Render one-page-summary.md and blog.md to print-ready HTML, then to PDF via headless Chrome/Edge.

Requires: pip install markdown   (Chrome or Edge must be installed for --print-to-pdf)
Run from the repo root:  python build/make_pdfs.py
"""
import markdown, subprocess, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; max-width: 100%; margin: 0; }
h1 { font-size: 18pt; line-height: 1.2; margin: 0 0 .3em; font-family: "Segoe UI", Helvetica, Arial, sans-serif; }
h2 { font-size: 13pt; margin: 1.1em 0 .35em; font-family: "Segoe UI", Helvetica, Arial, sans-serif;
     border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 11pt; margin: .9em 0 .3em; font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #333; }
p { margin: .5em 0; }
code { font-family: "Cascadia Code", Consolas, monospace; font-size: 9pt; background: #f0f0ee;
       padding: 0 2px; border-radius: 3px; }
pre { background: #f6f6f4; border: 1px solid #ddd; border-radius: 5px; padding: 8px 10px;
      font-size: 8.5pt; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: .8em 0; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eee; font-family: "Segoe UI", Helvetica, Arial, sans-serif; }
blockquote { border-left: 3px solid #888; margin: .8em 0; padding: .2em 0 .2em 12px; color: #444;
             font-style: italic; }
hr { border: 0; border-top: 1px solid #ccc; margin: 1.2em 0; }
strong { color: #000; }
a { color: #1a1a1a; text-decoration: underline; }
ul, ol { margin: .5em 0; padding-left: 22px; }
li { margin: .25em 0; }
"""


def build(md_name, pdf_name, title, compact=False):
    src = (ROOT / md_name).read_text(encoding="utf-8")
    html_body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
    extra = ""
    if compact:
        extra = ("@page{size:A4;margin:12mm 13mm;}"
                 "body{font-size:9.3pt;line-height:1.38;}"
                 "h1{font-size:15pt;margin-bottom:.2em;}"
                 "h2{font-size:11pt;margin:.7em 0 .25em;}"
                 "table{font-size:8pt;margin:.5em 0;}"
                 "th,td{padding:3px 5px;}"
                 "p{margin:.38em 0;}")
    page = (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{title}</title><style>{CSS}{extra}</style></head>'
            f'<body>{html_body}</body></html>')
    html_path = BUILD / pdf_name.replace(".pdf", ".html")
    html_path.write_text(page, encoding="utf-8")

    chrome = next((c for c in [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ] if pathlib.Path(c).exists()), None)
    if not chrome:
        print("No Chrome or Edge found for --print-to-pdf", file=sys.stderr)
        sys.exit(1)

    out = ROOT / pdf_name
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=90)
    print(f"{pdf_name}: {out.stat().st_size} bytes")


build("one-page-summary.md", "one-page-summary.pdf",
      "The Interference Budget - Concept Summary", compact=True)
build("blog.md", "blog.pdf", "The Interference Budget - Blog")
