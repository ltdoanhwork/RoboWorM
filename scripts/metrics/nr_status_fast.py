#!/usr/bin/env python3
"""Fast status for the dense-NR wave.

Uses a single os.scandir pass over results/ instead of glob: that directory now holds
~60k files on a parallel FS, where glob + per-file getmtime took >4 minutes while scandir
(which carries stat data with the dirent) takes seconds.
"""
import datetime
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R = os.path.join(REPO, 'metrics/WorldScore/results/dense_nr_full')
O = os.path.join(REPO, 'metrics/WorldScore/results/nr_opp')
now = time.time()

n5 = n10 = tot = 0
scored = set()
with os.scandir(f'{R}/results') as it:
    for e in it:
        if not e.name.endswith('.json'):
            continue
        tot += 1
        scored.add(e.name[:-5])
        age = now - e.stat().st_mtime
        if age < 300:
            n5 += 1
        if age < 600:
            n10 += 1
print(f'  5min: {n5:5d} clip ({n5/5:6.2f}/phut)')
print(f' 10min: {n10:5d} clip ({n10/10:6.2f}/phut)')
print(f' tong da cham: {tot}')

main = sum(1 for f in glob.glob(f'{R}/worker_*.log')
           if (now - os.path.getmtime(f)) / 60 < 8)
print(f'\nworker wave chinh: {main}')
opp = 0
for d in sorted(glob.glob(f'{O}/*/')):
    fs = glob.glob(f'{d}worker_*.log')
    if not fs:
        continue
    a = sum(1 for f in fs if (now - os.path.getmtime(f)) / 60 < 8)
    opp += a
    print(f'  {os.path.basename(d.rstrip("/")):18s} {a}/{len(fs)}')
print(f'worker ke: {opp}   TONG: {main + opp}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dense_nr import excluded_row          # noqa: E402

rows = json.load(open(f'{R}/manifest.json'))
ins = [x for x in rows if not excluded_row(x)]
left = len([x for x in ins if x['id'] not in scored])
rate = n10 / 10
h = left / rate / 60 if rate else 0
print(f'\nTIEN DO: {len(ins)-left}/{len(ins)} = {100*(len(ins)-left)/len(ins):.1f}%  con lai {left}')
if rate:
    # The containers run on UTC while the user reads times in UTC+7 (Asia/Ho_Chi_Minh),
    # so a bare datetime.now() printed an ETA seven hours in the past ("7.7h left ->
    # 17:11" when it was already 16:31 local). Print both, labelled.
    utc = datetime.datetime.now(datetime.timezone.utc)
    eta_utc = utc + datetime.timedelta(hours=h)
    vn = datetime.timezone(datetime.timedelta(hours=7))
    print(f'ETA @{rate:.1f}/phut: {h:.1f}h')
    print(f'  -> {eta_utc.astimezone(vn):%a %d/%m %H:%M} (gio VN, UTC+7)')
    print(f'  -> {eta_utc:%a %d/%m %H:%M} UTC')
print(f'stub {len(glob.glob(f"{R}/results/*.tmp"))}  errors {len(os.listdir(f"{R}/errors"))}')
