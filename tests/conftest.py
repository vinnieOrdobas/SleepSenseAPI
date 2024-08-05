import sys
import os

# Set the PYTHONPATH to include the root directory of your project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

print("PYTHONPATH:", sys.path)
