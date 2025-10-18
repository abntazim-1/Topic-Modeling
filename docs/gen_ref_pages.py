"""Script to generate API reference pages for MkDocs."""

import os
from pathlib import Path
from mkdocs_gen_files import Nav, Files

# Define the source directory
src_dir = Path("src")

# Create navigation structure
nav = Nav()

# Define the modules to document
modules = [
    "src.data",
    "src.features", 
    "src.topics",
    "src.models",
    "src.utils"
]

# Generate API reference pages
for module in modules:
    # Create the page content
    content = f"# {module.split('.')[-1].title()}\n\n::: {module}\n"
    
    # Write the page
    with open(f"docs/api/{module.split('.')[-1]}.md", "w") as f:
        f.write(content)
    
    # Add to navigation
    nav[module.split('.')[-1].title()] = f"api/{module.split('.')[-1]}.md"

# Write the navigation
with open("docs/api/index.md", "w") as f:
    f.write("# API Reference\n\n")
    f.write("This section contains the complete API documentation for the Topic Modeling Project, automatically generated from Python docstrings.\n\n")
    f.write("## Modules\n\n")
    for module in modules:
        f.write(f"::: {module}\n")

print("API reference pages generated successfully!")
