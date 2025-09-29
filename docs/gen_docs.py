import os
import subprocess
from pathlib import Path
from nbconvert import MarkdownExporter

DOCS_DIR = Path(__file__).parent
EXAMPLES_DIR = DOCS_DIR / "examples"
NOTEBOOKS_DIR = Path("examples")  # Update this to your notebooks path

def compile_cython_module():
    """Compile the Cython module for documentation generation"""
    import sys
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    
    extensions = [
        Extension("SuchTree.MuchTree",
                  ["SuchTree/MuchTree.pyx"],
                  extra_compile_args=["-O3"])
    ]
    
    # Build in a temporary directory and add to Python path
    build_dir = "build/temp"
    setup(
        name="SuchTree",
        ext_modules=cythonize(extensions, language_level="3"),
        script_args=["build_ext", "--build-temp", build_dir]
    )
    sys.path.insert(0, build_dir)
    return build_dir

def generate_api_docs():
    """Generate API documentation using compiled module"""
    # Ensure the module is compiled first and get build directory
    build_dir = compile_cython_module()
    
    # Import the compiled module for mkdocstrings
    import SuchTree.MuchTree

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
