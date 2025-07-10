import subprocess
import os

os.chdir('.')

# Parse all notebooks

notebooks = [f for f in os.listdir('.') if os.path.isfile(f)]
notebooks = filter(lambda s: s.split('_')[0].isdigit(), notebooks)
notebooks = sorted(notebooks, key=lambda s: int(s.split('_')[0]))

notebooks = notebooks[1:-1]

# Run notebooks
for nb in notebooks:
    print(f'Running {nb}...')
    subprocess.run([
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', nb
    ], check=True)