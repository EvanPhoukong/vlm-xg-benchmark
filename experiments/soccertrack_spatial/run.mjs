// Budgeted annotation-assisted visual comparison. No requests on import.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const OUT=path.join(ROOT,'results/soccertrack_spatial_v1');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const hash=b=>createHash('sha256').update(b).digest('hex');
const sha=async p=>hash(await fs.readFile(p));
async function optional(p,f){try{return await read(p);}catch(e){if(e.code==='ENOENT')return f;throw e;}}
async function save(p,x){
  await fs.mkdir(path.dirname(p),{recursive:true});const tmp=p+'.tmp-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
  await fs.writeFile(tmp,JSON.stringify(x,null,2));
  for(let i=0;i<8;i++)try{await fs.rename(tmp,p);return;}catch(e){if(!['EPERM','EBUSY','EACCES'].includes(e.code)||i===7)throw e;await new Promise(r=>setTimeout(r,150));}
}
async function key(){const t=await fs.readFile(path.join(ROOT,'../keys.txt'),'utf8');const a=[...new Set(t.match(/sk-ant-[A-Za-z0-9_-]+/g)||[])];if(a.length!==1)throw Error('Unique configured provider key required');return a[0];}
async function api(endpoint,token,body){
  const r=await fetch('https://api.anthropic.com/v1/'+endpoint,{method:'POST',headers:{'x-api-key':token,'anthropic-version':'2023-06-01','Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(150000)});
  const raw=await r.json();if(!r.ok){const e=Error('API HTTP'+r.status+' '+(raw.error?.type||'unknown')+'; no automatic retry');e.body=raw;e.http_status=r.status;throw e;}return raw;
}
export function cost(u,conservative=false){
  const ni=u.input_tokens,no=u.output_tokens,nw=u.cache_creation_input_tokens||0,nr=u.cache_read_input_tokens||0;
  if(![ni,no,nw,nr].every(Number.isInteger)||[ni,no,nw,nr].some(x=>x<0))throw Error('Invalid usage');
  return (conservative?(ni+nw+nr)*2.5+no*10:ni*2+nw*2.5+nr*.1+no*10)/1e6;
}
export function exposure(ledger){return ledger.reduce((s,e)=>s+(e.state==='completed'?e.conservative_usd:e.reserved_usd),0);}
export function canReserve(ledger,entry){if(ledger.some(e=>e.state==='uncertain'))throw Error('Uncertain prior charge');return ledger.length<1500&&exposure(ledger)+entry.reserved_usd<=10;}
export async function payload(job,p){
  const content=[];
  for(const item of job.images){const file=path.resolve(ROOT,item.path);if(!file.startsWith(ROOT+path.sep))throw Error('Image escapes project');const bytes=await fs.readFile(file);if(hash(bytes)!==item.sha256)throw Error('Image changed');content.push({type:'image',source:{type:'base64',media_type:item.path.endsWith('.jpg')?'image/jpeg':'image/png',data:bytes.toString('base64')}});}
  content.push({type:'text',text:p.prompt});
  const schema={type:'object',properties:{xg:{type:'number'},reason:{type:'string'}},required:['xg','reason'],additionalProperties:false};
  return {model:p.model,max_tokens:1024,inference_geo:'global',output_config:{effort:'low',format:{type:'json_schema',schema}},messages:[{role:'user',content}]};
}
export function validate(raw,p){
  if(!(raw.model===p.model||raw.model?.startsWith(p.model+'-'))||raw.type!=='message'||raw.stop_reason!=='end_turn')throw Error('Unexpected or incomplete model response');
  const answer=JSON.parse(raw.content.filter(x=>x.type==='text').map(x=>x.text).join(''));
  if(Object.keys(answer).sort().join(',')!=='reason,xg'||typeof answer.reason!=='string'||typeof answer.xg!=='number'||!Number.isFinite(answer.xg)||answer.xg<0||answer.xg>1)throw Error('Invalid raw probability');cost(raw.usage);return answer;
}
export async function checkFrozen(){
  const p=await read(path.join(OUT,'protocol.json'));
  for(const [field,name] of [['eligible_sha256','eligible.json'],['source_audit_sha256','source_audit.json'],['crop_audit_sha256','crop_audit.json'],['baseline_models_sha256','baseline_models.json']])if(await sha(path.join(OUT,name))!==p[field])throw Error('Frozen artifact changed');
  for(const [field,name] of [['python_sha256','experiment.py'],['common_sha256','common.py'],['source_runner_sha256','source.mjs'],['runner_sha256','run.mjs']])if(await sha(path.join(ROOT,'experiments/soccertrack_spatial',name))!==p[field])throw Error('Frozen source changed');
  return p;
}
function job(row,arm){const images=arm==='photo_diagram'?[row.images.raw,row.images.diagram]:[row.images[arm]];return {shot_id:row.shot_id,match_id:row.match_id,arm,images};}
export async function setup(){
  const p=await checkFrozen(),rows=await read(path.join(OUT,'eligible.json')),token=await key(),mids=[...new Set(rows.map(r=>r.match_id))].sort(),counts={};
  const jobs=[];
  for(const mid of mids)for(const arm of p.arms){
    const candidates=rows.filter(r=>r.match_id===mid);let largest=null,area=-1;
    for(const row of candidates){let bytes=0;for(const image of job(row,arm).images)bytes+=(await fs.stat(path.join(ROOT,image.path))).size;if(bytes>area){area=bytes;largest=row;}}
    jobs.push(job(largest,arm));
  }
  // File size selects preview candidates; add a global1024-token bound and verify all actual usage.
  for(let i=0;i<jobs.length;i+=6){await Promise.all(jobs.slice(i,i+6).map(async j=>{const body=await payload(j,p);const n=await api('messages/count_tokens',token,{model:body.model,messages:body.messages});if(!Number.isInteger(n.input_tokens)||n.input_tokens<=0)throw Error('Input count absent');counts[j.match_id+':'+j.arm]=n.input_tokens;}));await save(path.join(OUT,'token_preview.json'),counts);}
  const bound=Math.max(...Object.values(counts))+1024;
  const candidates=[...new Set([rows.length,240,200,150,120,100,80].filter(n=>n<=rows.length))].sort((a,b)=>b-a),forecasts={};
  for(const n of candidates){const expected=rows.slice(0,n).reduce((sum,r)=>sum+p.arms.reduce((s,a)=>s+((counts[r.match_id+':'+a]+256)*2.5+128*10)/1e6,0),0);forecasts[n]=1.15*expected+6*(bound*2.5+1024*10)/1e6+.20;}
  const n=candidates.find(n=>forecasts[n]<=10);if(!n)throw Error('No planned cohort fits$10');const selected=rows.slice(0,n),schedule=[];
  // Deterministic arm order independent of outcomes; cyclic rotations balance condition order.
  for(let i=0;i<selected.length;i++){const order=p.arms.slice(i%4).concat(p.arms.slice(0,i%4));schedule.push(...order.map(a=>job(selected[i],a)));}
  await save(path.join(OUT,'evaluation/selection.json'),selected);await save(path.join(OUT,'evaluation/jobs.json'),schedule);
  const selection={n,jobs:schedule.length,forecasts_usd:forecasts,input_bound:bound,selection_sha256:await sha(path.join(OUT,'evaluation/selection.json')),jobs_sha256:await sha(path.join(OUT,'evaluation/jobs.json')),preview_sha256:await sha(path.join(OUT,'token_preview.json')),rule:'Pre-generation usage-only forecast;all four conditions fixed.'};
  await save(path.join(OUT,'resource_choice.json'),selection);return selection;
}
export async function status(){const ledger=await optional(path.join(OUT,'attempts.json'),[]),choice=await read(path.join(OUT,'resource_choice.json'));return {completed:ledger.filter(e=>e.state==='completed').length,attempts:ledger.length,planned:choice.jobs,estimated_usd:ledger.reduce((s,e)=>s+(e.estimated_usd||0),0),conservative_exposure_usd:exposure(ledger),uncertain:ledger.filter(e=>e.state!=='completed').length,cap_usd:10};}
export async function runBatch(limit=48){
  const p=await checkFrozen(),choice=await read(path.join(OUT,'resource_choice.json')),jobs=await read(path.join(OUT,'evaluation/jobs.json')),ledger=await optional(path.join(OUT,'attempts.json'),[]),token=await key();
  if(await sha(path.join(OUT,'evaluation/jobs.json'))!==choice.jobs_sha256||await sha(path.join(OUT,'token_preview.json'))!==choice.preview_sha256)throw Error('Frozen schedule/bounds changed');
  if(ledger.some(e=>e.state!=='completed'))throw Error('Unresolved prior request;stop');if(JSON.stringify(ledger.map(e=>e.job))!==JSON.stringify(jobs.slice(0,ledger.length)))throw Error('Schedule prefix changed');
  const selected=jobs.slice(ledger.length,ledger.length+limit),lock=await fs.open(path.join(OUT,'generation.lock'),'wx');let serial=Promise.resolve(),stopped=false;
  const commit=fn=>{const next=serial.then(fn);serial=next.catch(()=>{});return next;};
  try{
    for(let i=0;i<selected.length&&!stopped;i+=6){
      const prepared=await Promise.all(selected.slice(i,i+6).map(async j=>{const body=await payload(j,p);return {body,entry:{job:j,state:'reserved',input_bound:choice.input_bound,reserved_usd:(choice.input_bound*2.5+1024*10)/1e6,request_sha256:hash(JSON.stringify(body)),response_file:'evaluation/responses/'+j.shot_id+'_'+j.arm+'.json',started_utc:new Date().toISOString()}};}));
      const temp=ledger.slice();for(const {entry} of prepared){if(!canReserve(temp,entry))throw Error('Hard$10 cap reached before dispatch');temp.push(entry);}ledger.push(...prepared.map(x=>x.entry));await save(path.join(OUT,'attempts.json'),ledger);
      await Promise.all(prepared.map(async({body,entry})=>{try{const raw=await api('messages',token,body);await save(path.join(OUT,entry.response_file),raw);const answer=validate(raw,p);const ni=raw.usage.input_tokens+(raw.usage.cache_creation_input_tokens||0)+(raw.usage.cache_read_input_tokens||0);if(ni>entry.input_bound||raw.usage.output_tokens>1024)throw Error('Reserved tokens exceeded');Object.assign(entry,{state:'completed',answer,usage:raw.usage,estimated_usd:cost(raw.usage),conservative_usd:cost(raw.usage,true),returned_model:raw.model,completed_utc:new Date().toISOString()});}catch(e){Object.assign(entry,{state:'uncertain',error:e.message,error_body:e.body,http_status:e.http_status});stopped=true;}await commit(()=>save(path.join(OUT,'attempts.json'),ledger));}));
      await save(path.join(OUT,'status.json'),await status());
    }
    const result=await status();if(result.completed===jobs.length)await save(path.join(OUT,'complete.json'),result);if(stopped)throw Error('Stopped after unresolved request;no automatic retry');return result;
  }finally{await lock.close();await fs.unlink(path.join(OUT,'generation.lock'));}
}
