"""Run a command under a hard memory cap (macOS has no working RLIMIT_AS).

Polls the total RSS of the command's whole process tree and kills the tree if it
exceeds the cap. Prints the peak on exit.

    python tools/memguard.py --cap-gb 4 -- uv run python experiment/run_arms.py ...
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def tree_rss_bytes(root_pid):
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children, rss = {}, {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, kb = map(int, parts)
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb * 1024
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += children.get(p, [])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    cap = args.cap_gb * 1e9
    proc = subprocess.Popen(cmd, start_new_session=True)
    peak = 0
    while proc.poll() is None:
        rss = tree_rss_bytes(proc.pid)
        peak = max(peak, rss)
        if rss > cap:
            print(f"[memguard] RSS {rss / 1e9:.2f} GB > cap {args.cap_gb} GB: killing {cmd[:4]}",
                  file=sys.stderr, flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"[memguard] peak {peak / 1e9:.2f} GB (killed)", file=sys.stderr)
            sys.exit(137)
        time.sleep(args.interval)
    print(f"[memguard] peak {peak / 1e9:.2f} GB, exit {proc.returncode}", file=sys.stderr)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
