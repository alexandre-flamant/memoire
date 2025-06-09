import subprocess

notebooks = ['nb.ipynb', 'carnb.ipynb', 'testnb.ipynb']

for nb in notebooks:
    print(f'Running {nb}...')
    subprocess.run([
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--inplace', nb
    ], check=True)
