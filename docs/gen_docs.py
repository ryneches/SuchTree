import os
import sys
import subprocess
from pathlib import Path
from nbconvert import MarkdownExporter

DOCS_DIR      = Path(__file__).parent
EXAMPLES_DIR  = Path( os.path.join( DOCS_DIR, 'examples' ) )
NOTEBOOKS_DIR = EXAMPLES_DIR # Update this to your notebooks path
PROJECT_ROOT  = DOCS_DIR.parent

def build_package_inplace() :
    '''Build the package in place using its existing setup configuration'''
    print( 'Building package in place with existing Cython configuration...' )
    
    # Change to project root directory
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    
    try:
        # Build extension modules in place without installing
        # This uses the existing setup.py/pyproject.toml configuration
        result = subprocess.run(
            [ sys.executable, 'setup.py', 'build_ext', '--inplace' ],
            check          = True,
            capture_output = True,
            text           = True
        )
        print( 'Package built in place successfully' )
        print( result.stdout )
        
    except subprocess.CalledProcessError as e :
        print( f'Error building package: {e}' )
        print( f'stdout: {e.stdout}' )
        print( f'stderr: {e.stderr}' )
        raise
    finally :
        os.chdir( original_dir )

def generate_api_docs():
    '''Generate API documentation using built package'''
    # Ensure the package is built first
    build_package_inplace()
    
    # The module should now be importable from the built package
    try :
        import SuchTree
        print( f'Successfully imported SuchTree from {SuchTree.__file__}' )
    except ImportError as e :
        print( f'Warning: Could not import SuchTree : {e}' )
        print( 'API documentation generation may be incomplete' )

def convert_notebooks() :
    '''Convert Jupyter notebooks to Markdown'''
    md_exporter = MarkdownExporter()
    os.makedirs( EXAMPLES_DIR, exist_ok=True )
    
    notebook_count = 0
    for nb_path in NOTEBOOKS_DIR.glob( '*.ipynb' ) :
        output_path = os.path.join( EXAMPLES_DIR, f'{nb_path.stem}.md' )
        print( f'Converting {nb_path} to {output_path}' )
        body, _ = md_exporter.from_filename( nb_path )
        with open(output_path, 'w' ) as f :
            f.write( f'# {nb_path.stem}\n\n' )
            f.write( body )
        notebook_count += 1
    
    print( f'Converted {notebook_count} notebook(s) to Markdown' )

if __name__ == '__main__' :
    generate_api_docs()
    convert_notebooks()
    print( 'Documentation generated successfully!' )
