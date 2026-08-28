"""Thread-cap wrapper: pin torch intra+inter-op to 1 BEFORE the target runs, then exec it.
Fixes futex contention from torch's node-wide (e.g. 96) interop pool on a 2-core cgroup alloc,
which otherwise pins the process at ~7% CPU. Usage: python run_capped.py <target.py> [args...]."""
import sys, runpy, torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
