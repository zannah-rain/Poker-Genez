import os
import sys

POKER_GA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poker_ga"))
if POKER_GA_DIR not in sys.path:
    sys.path.insert(0, POKER_GA_DIR)
