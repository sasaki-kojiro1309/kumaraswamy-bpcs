"""External 23-minute cap in addition to the shared 25/30-minute deadlines."""
import os
import argparse
from pathlib import Path
import signal
import subprocess
import time
from .simulation_storage import read_json,commit_json


def children(parent):
    lines=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True).splitlines()
    return [int(line.split()[0]) for line in lines if len(line.split())==2 and int(line.split()[1])==parent]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',type=int,required=True)
    args=parser.parse_args()
    root=Path('results/simulation_submission')
    logs=sorted(root.glob('diagnostics/priority_supervision/*/started.json'))[-1].parent
    start,_=read_json(logs/'started.json')
    # The supervisor began before the full two-minute inspection allowance was
    # used. This stricter cap enforces <=23 minutes of fitting independently.
    cap_unix=min(start['compute_deadline_unix'],start['supervisor_started_unix']+1380)
    cap=time.monotonic()+cap_unix-time.time()
    parent=args.parent_pid
    commit_json(logs/'external_cap_started.json',dict(parent_pid=parent,cap_monotonic=cap,
        cap_unix=cap_unix,reason='At most 23 minutes from supervisor launch, with 8-second checkpoint reserve.'))
    while time.monotonic()<cap-8 and not (logs/'finished.json').exists():time.sleep(.5)
    signaled=[]
    if not (logs/'finished.json').exists():
        for pid in children(parent):
            try:os.kill(pid,signal.SIGALRM);signaled.append(pid)
            except ProcessLookupError:pass
        while time.monotonic()<cap and children(parent):time.sleep(.2)
        for pid in children(parent):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    commit_json(logs/'external_cap_finished.json',dict(finished_unix=time.time(),
        alarmed_workers=signaled,remaining_child_pids=children(parent)))


if __name__=='__main__':main()
