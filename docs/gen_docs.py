import os
import subprocess
from pathlib import Path
from nbconvert import MarkdownExporter

DOCS_DIR = Path(__file__).parent
EXAMPLES_DIR = DOCS_DIR / "examples"
NOTEBOOKS_DIR = Path("examples")  # Update this to your notebooks path

def compile_cython_module():
    """Compile the Cython module for documentation generation"""
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    
    extensions = [
        Extension("SuchTree.MuchTree",
                  ["SuchTree/MuchTree.pyx"],
                  extra_compile_args=["-O3"])
    ]
    
    setup(
        name="SuchTree",
        ext_modules=cythonize(extensions, language_level="3"),
        script_args=["build_ext", "--inplace"]
    )

def generate_api_docs():
    """Generate API documentation using compiled module"""
    # Ensure the module is compiled first
    compile_cython_module()

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
