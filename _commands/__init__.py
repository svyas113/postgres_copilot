# This file makes the _commands directory a proper Python package
# This helps with relative imports in the fastworkflow application
import sys
import os

# Add the parent directory to the path so we can import modules from the application directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
