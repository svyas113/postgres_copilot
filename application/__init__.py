# This file makes the application directory a proper Python package
# This helps with relative imports in the fastworkflow application
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
