import os

# Setup a memoire directory variable
memoire_dir = os.environ['memoire']
runs_dir = rf"{memoire_dir}/05-results"

codes = [
    100 * type + 10 * ds + loss
    for type in (0, 1)
    for ds in (0, 1, 2)
    for loss in (1, 2, 3, 4)
]

DIRECTORY = {f"{code:03d}": rf"{runs_dir}/{code:03d}" for code in codes}
PORT = {f"{code:03d}": 5000 + code for code in codes}
