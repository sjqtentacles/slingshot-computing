"""Inject out/switch.json into the viewer template -> out/viewer.html."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    data = (ROOT / "out" / "switch.json").read_text()
    template = (ROOT / "demos" / "viewer.html").read_text()
    assert "/*__DATA__*/" in template, "placeholder missing from template"
    out = ROOT / "out" / "viewer.html"
    out.write_text(template.replace("/*__DATA__*/", data, 1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
