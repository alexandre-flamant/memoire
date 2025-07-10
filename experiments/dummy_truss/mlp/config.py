import os

# Setup a memoire directory variable
memoire_dir = os.environ['memoire']
runs_dir = rf"{memoire_dir}/05-results/dummy"

codes = [1,2,3,4,5]

DIRECTORY = {f"{code:03d}": rf"{runs_dir}/{code:03d}" for code in codes}
PORT = {f"{code:03d}": 6000 + code for code in codes}