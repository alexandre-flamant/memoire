import os

# Setup a memoire directory variable
memoire_dir = os.environ['memoire']
runs_dir = rf"{memoire_dir}/05-results"

codes = (1, 2, 3, 11, 12, 13, 21, 22, 23, 101, 102, 103, 111, 112, 113, 121, 122, 123)

DIRECTORY = {f"{code:03d}": rf"{runs_dir}/{code:03d}" for code in codes}
PORT = {f"{code:03d}": 5000 + code for code in codes}