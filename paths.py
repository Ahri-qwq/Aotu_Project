import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BAND_DIR = os.path.join(DATA_DIR, "bandstructures")
CASE_DIR = os.path.join(DATA_DIR, "case_outputs")
MPID_FILE = os.path.join(DATA_DIR, "mpids_with_bandstructure.txt")

os.makedirs(BAND_DIR, exist_ok=True)
os.makedirs(CASE_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Band Dir: {BAND_DIR}")
