import os
import sys
import glob
import shutil
from pathlib import Path


def build_index_html(dest_dir: Path, titles: list[str], files: list[str]):
    items = []
    for title, fname in zip(titles, files):
        items.append(f"<li><a href=\"{os.path.basename(fname)}\">{title}</a></li>")
    index_html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Documentation</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.8rem; }}
    ul {{ line-height: 1.8; }}
    .note {{ background:#f6f8fa; padding:0.75rem 1rem; border-radius:6px; margin:1rem 0; }}
  </style>
  <link rel="icon" href="data:,">
  </head>
<body>
  <h1>Topic Modeling Project – Documentation</h1>
  <p class="note">This index links to generated documentation pages from Markdown sources.</p>
  <ul>
    {''.join(items)}
  </ul>
</body>
</html>
"""
    (dest_dir / "index.html").write_text(index_html, encoding="utf-8")


def convert_markdown_to_html(md_path: Path, html_path: Path):
    try:
        import markdown  # type: ignore
    except Exception:
        # Fallback: wrap Markdown in <pre> when markdown lib is missing
        content = md_path.read_text(encoding="utf-8")
        html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{md_path.name}</title>
  <style>pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono','Courier New', monospace; }}</style>
</head>
<body>
  <pre>{content}</pre>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        return False

    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["fenced_code", "tables", "toc"])  # pragma: allowlist ok
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{md_path.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; }}
    pre, code {{ background:#f6f8fa; padding:0.2rem 0.4rem; border-radius:4px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border:1px solid #ddd; padding: 6px 8px; }}
  </style>
</head>
<body>
  {html_body}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build project documentation from Markdown sources")
    parser.add_argument("--src", default="docs", help="Source docs directory")
    parser.add_argument("--out", default="docs/_build", help="Output build directory")
    parser.add_argument("--format", choices=["html", "md"], default="html", help="Build output format")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = [Path(p) for p in glob.glob(str(src_dir / "**/*.md"), recursive=True)]
    # Optionally include root README.md in documentation build
    root_readme = Path("README.md")
    include_readme = root_readme.exists()
    if not md_files:
        print(f"No Markdown files found under {src_dir}")
        return 0

    generated_files = []
    titles = []
    used_md_to_html = False

    if args.format == "md":
        # Copy raw Markdown files into build dir
        for md in md_files:
            dest = out_dir / md.name
            shutil.copy(md, dest)
            generated_files.append(str(dest))
            titles.append(md.stem.replace('_', ' ').title())
        if include_readme:
            dest = out_dir / root_readme.name
            shutil.copy(root_readme, dest)
            generated_files.append(str(dest))
            titles.append("Project README")
        build_index_html(out_dir, titles, generated_files)
        copied = len(md_files) + (1 if include_readme else 0)
        print(f"Copied {copied} Markdown files into {out_dir}")
        return 0

    # HTML build path
    for md in md_files:
        dest = out_dir / (md.stem + ".html")
        used = convert_markdown_to_html(md, dest)
        used_md_to_html = used_md_to_html or used
        generated_files.append(str(dest))
        titles.append(md.stem.replace('_', ' ').title())
    if include_readme:
        dest = out_dir / "README.html"
        used = convert_markdown_to_html(root_readme, dest)
        used_md_to_html = used_md_to_html or used
        generated_files.append(str(dest))
        titles.append("Project README")

    build_index_html(out_dir, titles, generated_files)
    total_pages = len(md_files) + (1 if include_readme else 0)
    msg = f"Built HTML docs ({total_pages} pages) into {out_dir}"
    if used_md_to_html:
        msg += " using Python markdown conversion"
    else:
        msg += " (markdown lib not available; used plain <pre> fallback)"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())