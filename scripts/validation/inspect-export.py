"""Validate gzip/annotated CSV and inventory each raw field observation."""
import csv, gzip, json, hashlib, sys
from pathlib import Path
from collections import Counter
root=Path(sys.argv[1]); manifest=json.loads((root/'manifest.json').read_text())
server_counts={r['bucket']:r['server_rows'] for r in json.loads((root/'server-counts.json').read_text())} if (root/'server-counts.json').exists() else {}
for source in manifest['sources']:
    if source['status'] not in ('downloaded','verified'): continue
    file=root/source['file']; counts=Counter(); fields=Counter(); devices=set(); headers=None; n=0; first=last=None; columns=set()
    with gzip.open(file,'rt',newline='') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'): continue
            if '_time' in row and '_value' in row:
                headers=row; columns.update(row); continue
            if 'error' in row and 'reference' in row: raise RuntimeError(f'Influx query error in {file}')
            if headers is None: raise RuntimeError(f'Unexpected CSV in {file}')
            if len(row)!=len(headers): raise RuntimeError(f'Invalid row width in {file}')
            r=dict(zip(headers,row)); t=r['_time']; n+=1
            first=t if first is None or t<first else first; last=t if last is None or t>last else last
            counts[r.get('_measurement','')]+=1; fields[(r.get('_measurement',''),r.get('_field',''))]+=1
            if r.get('devEui'): devices.add(r['devEui'])
    if source['bucket'] in server_counts and n != server_counts[source['bucket']]: raise RuntimeError(f'Server/local count mismatch: {file}')
    summary={'rows':n,'earliest_utc':first,'latest_utc':last,'measurements':dict(counts),'fields':[{'measurement':m,'field':f,'rows':c} for (m,f),c in sorted(fields.items())],'device_euis':sorted(devices),'columns':sorted(columns)}
    (root/(file.name.removesuffix('.csv.gz')+'-summary.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    digest=hashlib.sha256()
    with file.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    source.update(status='verified',rows=n,earliest_utc=first,latest_utc=last,sha256=digest.hexdigest())
    print(json.dumps({'bucket':source['bucket'],**{k:summary[k] for k in ['rows','earliest_utc','latest_utc','measurements','columns']}}))
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
