"""Inject simulation JSON into the viewer templates -> out/*.html."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

PAGES = [
    ("demos/viewer.html", "out/switch.json", "out/viewer.html"),
    ("demos/arithmetic.html", "out/arithmetic.json", "out/arithmetic.html"),
    ("demos/pipeline.html", "out/pipeline.json", "out/pipeline.html"),
]


def main():
    for template_path, data_path, out_path in PAGES:
        data = (ROOT / data_path).read_text()
        template = (ROOT / template_path).read_text()
        assert "/*__DATA__*/" in template, f"placeholder missing from {template_path}"
        out = ROOT / out_path
        out.write_text(template.replace("/*__DATA__*/", data, 1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
