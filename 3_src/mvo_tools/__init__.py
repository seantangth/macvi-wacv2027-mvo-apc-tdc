"""MVO (Maritime Visual Odometry) team tooling. Pure NumPy; the starter is vendored in 3_src/starter_upstream."""
import os, sys
_STARTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'starter_upstream', 'starter')
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)
