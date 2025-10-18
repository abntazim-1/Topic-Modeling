PY := python
DOCS_SRC := docs
DOCS_BUILD := $(DOCS_SRC)/_build

.PHONY: docs docs-md docs-clean docs-open docs-serve

## Build documentation into docs/_build (HTML preferred; falls back if markdown lib missing)
docs:
	$(PY) tools/build_docs.py --src $(DOCS_SRC) --out $(DOCS_BUILD) --format html

## Copy raw Markdown files into docs/_build
docs-md:
	$(PY) tools/build_docs.py --src $(DOCS_SRC) --out $(DOCS_BUILD) --format md

## Remove built documentation
docs-clean:
	$(PY) -c "import shutil, sys; shutil.rmtree('$(DOCS_BUILD)', ignore_errors=True); print('Removed $(DOCS_BUILD)')"

## Open built documentation index in the default browser
docs-open:
	$(PY) -c "import webbrowser, os; p=os.path.abspath('$(DOCS_BUILD)/index.html'); print('Opening', p); webbrowser.open('file://'+p)"

## Serve built documentation locally on http://localhost:8000/
docs-serve:
	$(PY) -m http.server 8000 --directory $(DOCS_BUILD)