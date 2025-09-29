import os
import subprocess
from pathlib import Path
from nbconvert import MarkdownExporter

DOCS_DIR = Path(__file__).parent
EXAMPLES_DIR = DOCS_DIR / "examples"
NOTEBOOKS_DIR = Path("examples")  # Update this to your notebooks path

def generate_api_docs():
    """Generate API documentation using pydoc"""
    os.makedirs(DOCS_DIR / "api", exist_ok=True)
    subprocess.run([
        "pydoc", "-w", 
        "SuchTree.SuchTree",
        "SuchTree.SuchLinkedTrees",
        "SuchTree.exceptions"
    ], cwd=DOCS_DIR / "api")

def convert_notebooks():
    """Convert Jupyter notebooks to Markdown"""
    md_exporter = MarkdownExporter()
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    for nb_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        output_path = EXAMPLES_DIR / f"{nb_path.stem}.md"
        body, _ = md_exporter.from_filename(nb_path)
        with open(output_path, "w") as f:
            f.write(f"# {nb_path.stem}\n\n")
            f.write(body)

if __name__ == "__main__":
    generate_api_docs()
    convert_notebooks()
    print("Documentation generated successfully!")
