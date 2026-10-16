import fs from 'node:fs';
import path from 'node:path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'node:url';
const toolRoot = fileURLToPath(new URL('../', import.meta.url));
const projectRoot = path.resolve(toolRoot, '..');
import { InfluxDB } from '@influxdata/influxdb-client';
dotenv.config({path:process.env.ZB202_ENV_FILE||path.join(toolRoot, '.env'),quiet:true});
if (!process.argv[2]) throw new Error('Usage: npm run data:verify -- <snapshot-path-relative-to-project-root>');
const root=path.resolve(projectRoot, process.argv[2]), m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'))), e=process.env;
const counts=[];
for(const s of m.sources){
 if(s.bucket!=='zb202_iot') throw new Error('Only zb202_iot is supported');
 const token=e.ZB202_INFLUX_TOKEN;
 const api=new InfluxDB({url:e.ZB202_INFLUX_URL,token,timeout:600000}).getQueryApi(e.ZB202_INFLUX_ORG);
 const rows=await api.collectRows(`from(bucket: ${JSON.stringify(s.bucket)}) |> range(start: time(v: ${JSON.stringify(m.start)}), stop: time(v: ${JSON.stringify(m.stop)})) |> count() |> keep(columns: ["_value"]) |> group() |> sum()`);
 counts.push({bucket:s.bucket,server_rows:rows[0]?._value||0}); console.log(counts.at(-1));
}
fs.writeFileSync(path.join(root,'server-counts.json'),JSON.stringify(counts,null,2)+'\n');
