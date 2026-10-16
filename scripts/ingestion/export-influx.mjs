import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createGzip } from 'node:zlib';
import dotenv from 'dotenv';
dotenv.config({path: process.env.ZB202_ENV_FILE || '.env', quiet:true});
const e=process.env, q=JSON.stringify;
const root=path.resolve('data','raw','zb202_iot',new Date().toISOString().replaceAll(':','-'));
fs.mkdirSync(root,{recursive:true});
const stop=new Date().toISOString();
const sources=[{bucket:'zb202_iot',token:e.ZB202_INFLUX_TOKEN}];
if (!e.ZB202_INFLUX_URL || !e.ZB202_INFLUX_ORG || !e.ZB202_INFLUX_TOKEN) throw new Error('Set ZB202_INFLUX_URL, ZB202_INFLUX_ORG and ZB202_INFLUX_TOKEN in .env');
const manifest={url:e.ZB202_INFLUX_URL,org:e.ZB202_INFLUX_ORG,start:'1970-01-01T00:00:00Z',stop,format:'InfluxDB annotated CSV, gzip compressed; original field values and tags; no aggregation',sources:[]};
function save(){fs.writeFileSync(path.join(root,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');}
async function query(source,flux){const r=await fetch(new URL('/api/v2/query?org='+encodeURIComponent(e.ZB202_INFLUX_ORG),e.ZB202_INFLUX_URL),{method:'POST',headers:{Authorization:'Token '+source.token,'Content-Type':'application/json',Accept:'text/csv'},body:JSON.stringify({query:flux,type:'flux',dialect:{header:true,delimiter:',',annotations:['datatype','group','default'],commentPrefix:'#',dateTimeFormat:'RFC3339Nano'}}),signal:AbortSignal.timeout(600000)});if(!r.ok)throw Error('HTTP '+r.status+' '+(await r.text()).slice(0,1000));return r;}
for(const source of sources){const entry={bucket:source.bucket,status:'started'};manifest.sources.push(entry);save();try{
 const base=`from(bucket: ${q(source.bucket)}) |> range(start: time(v: ${q(manifest.start)}), stop: time(v: ${q(stop)}))`;
 const bounds=await query(source,base+' |> keep(columns: ["_time"]) |> group() |> min(column: "_time")');
 const earliest=await bounds.text();fs.writeFileSync(path.join(root,source.bucket.replace(/[^\w.-]/g,'_')+'-earliest.csv'),earliest);console.log(source.bucket+' earliest: '+earliest.trim());
 const name=source.bucket.replace(/[^\w.-]/g,'_')+'.csv.gz';
 const r=await query(source,base);await pipeline(Readable.fromWeb(r.body),createGzip(),fs.createWriteStream(path.join(root,name+'.partial')));
 fs.renameSync(path.join(root,name+'.partial'),path.join(root,name));Object.assign(entry,{status:'downloaded',file:name,bytes:fs.statSync(path.join(root,name)).size});console.log(source.bucket+' downloaded '+entry.bytes+' bytes');
 }catch(err){entry.status='failed';entry.error=err.message;console.log(source.bucket+' failed: '+err.message);}save();}
console.log('EXPORT_DIR='+root);
if(manifest.sources.some(x=>x.status==='failed'))process.exitCode=1;
