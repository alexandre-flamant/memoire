import subprocess
import os

os.chdir('./experiments/mlp')

# Parse all notebooks

notebooks = [f for f in os.listdir('.') if os.path.isfile(f)]
notebooks = filter(lambda s: s.split('_')[0].isdigit(), notebooks)
notebooks = filter(lambda s: int(s.split('_')[0]) in {113, 123}, notebooks)
notebooks = sorted(notebooks, key=lambda s: int(s.split('_')[0]))

# Run notebooks"
for nb in notebooks:
    print(f'Running {nb}...')
    subprocess.run([
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', nb
    ], check=True)
