// Five matched conditions. No network calls on import.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {validate} from '../soccertrack_spatial/run.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const OUT=path.join(ROOT,'results/soccertrack_spatial_icl_v1');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const hash=b=>createHash('sha256').update(b).digest('hex');
async function optional(p,f){try{return await read(p);}catch(e){if(e.code==='ENOENT')return f;throw e;}}
async function save(p,x){
  await fs.mkdir(path.dirname(p),{recursive:true});const tmp=p+'.tmp-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
  await fs.writeFile(tmp,JSON.stringify(x,null,2));
  for(let i=0;i<8;i++)try{await fs.rename(tmp,p);return;}catch(e){if(!['EPERM','EBUSY','EACCES'].includes(e.code)||i===7)throw e;await new Promise(r=>setTimeout(r,150));}
}
async function key(){const t=await fs.readFile(path.join(ROOT,'../keys.txt'),'utf8'),a=[...new Set(t.match(/sk-ant-[A-Za-z0-9_-]+/g)||[])];if(a.length!==1)throw Error('Unique configured provider key required');return a[0];}
async function api(endpoint,token,body){
  const r=await fetch('https://api.anthropic.com/v1/'+endpoint,{method:'POST',headers:{'x-api-key':token,'anthropic-version':'2023-06-01','Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(150000)});
  const raw=await r.json();if(!r.ok){const e=Error('API HTTP'+r.status+' '+(raw.error?.type||'unknown')+'; no automatic retry');e.body=raw;e.http_status=r.status;throw e;}return raw;
}
export function cost(u,conservative=false){
  const ni=u.input_tokens,no=u.output_tokens,nw=u.cache_creation_input_tokens||0,nr=u.cache_read_input_tokens||0;
  if(![ni,no,nw,nr].every(Number.isInteger)||[ni,no,nw,nr].some(n=>n<0))throw Error('Invalid usage');
  if((u.cache_creation?.ephemeral_1h_input_tokens||0)>0)throw Error('Unexpected1-hour cache charge');
  return (ni*(conservative?2.5:2)+nw*2.5+nr*(conservative?.125:.1)+no*10)/1e6;
}
export function exposure(ledger){return ledger.reduce((s,e)=>s+(e.state==='completed'?e.conservative_usd:e.state==='cancelled'?0:e.reserved_usd),0);}
export function blockSize(ledger,bound,remaining){
  if(ledger.some(e=>!['completed','cancelled'].includes(e.state)))throw Error('Unresolved prior request');
  const perShot=5*(bound*2.5+1024*10)/1e6;
  for(let n=Math.min(10,remaining);n>=1;n--)if(exposure(ledger)+perShot*n<=10)return n;
  return 0;
}
const cache=new Map();
async function image(item){
  if(cache.has(item.path))return structuredClone(cache.get(item.path));
  const file=path.resolve(ROOT,item.path);if(!file.startsWith(ROOT+path.sep))throw Error('Image escapes project');
  const bytes=await fs.readFile(file);if(hash(bytes)!==item.sha256)throw Error('Image changed');
  const block={type:'image',source:{type:'base64',media_type:item.path.endsWith('.jpg')?'image/jpeg':'image/png',data:bytes.toString('base64')}};
  cache.set(item.path,block);if(cache.size>400)cache.delete(cache.keys().next().value);return structuredClone(block);
}
export async function payload(job,p){
  const content=[];
  if(job.arm!=='zero'){
    content.push({type:'text',text:job.arm.includes('outcome')?p.outcome_intro:p.teacher_intro});
    for(let i=0;i<job.pages.length;i++){const block=await image(job.pages[i]);if(i===job.pages.length-1)block.cache_control={type:'ephemeral'};content.push(block);}
    content.push({type:'text',text:'NEW QUERY (outcome and reference probability unknown):'});
  }
  content.push(await image(job.query));content.push({type:'text',text:p.prompt});
  const schema={type:'object',properties:{xg:{type:'number'},reason:{type:'string'}},required:['xg','reason'],additionalProperties:false};
  return {model:p.model,max_tokens:1024,inference_geo:'global',output_config:{effort:'low',format:{type:'json_schema',schema}},messages:[{role:'user',content}]};
}
export async function checkFrozen(){
  const p=await read(path.join(OUT,'protocol.json'));
  for(const [relative,expected] of Object.entries(p.frozen))if(hash(await fs.readFile(path.join(ROOT,relative)))!==expected)throw Error('Frozen dependency changed:'+relative);
  return p;
}
function job(row,arm,pages,index){return {shot_id:row.shot_id,match_id:row.match_id,pool_index:index,arm,query:row.images.diagram,pages:pages[row.match_id][arm]};}
export async function setup(){
  if(await optional(path.join(OUT,'resource_choice.json'),null))throw Error('Resource plan already frozen');
  const p=await checkFrozen(),pool=await read(path.join(OUT,'pool.json')),pages=await read(path.join(OUT,'pages.json')),token=await key(),counts={},previews=[];
  for(const mid of [...new Set(pool.map(r=>r.match_id))].sort()){
    let chosen=null,bytes=-1;for(const row of pool.filter(r=>r.match_id===mid)){const n=(await fs.stat(path.join(ROOT,row.images.diagram.path))).size;if(n>bytes){bytes=n;chosen=row;}}
    for(const arm of p.arms)previews.push(job(chosen,arm,pages,pool.indexOf(chosen)));
  }
  for(let i=0;i<previews.length;i+=6){await Promise.all(previews.slice(i,i+6).map(async j=>{const body=await payload(j,p),n=await api('messages/count_tokens',token,{model:body.model,messages:body.messages});if(!Number.isInteger(n.input_tokens)||n.input_tokens<=0)throw Error('Count missing');counts[j.match_id+':'+j.arm]=n.input_tokens;}));await save(path.join(OUT,'token_preview.json'),counts);}
  const bound=Math.max(...Object.values(counts))+1024,perShot=5*(bound*2.5+1024*10)/1e6;
  const choice={input_bound:bound,pool_n:pool.length,arms:5,max_calls:pool.length*5,maximum_uncached_reservation_per_shot:perShot,
    preview_sha256:hash(await fs.readFile(path.join(OUT,'token_preview.json'))),rule:p.stopping_rule};
  await save(path.join(OUT,'resource_choice.json'),choice);return choice;
}
export async function status(){
  const ledger=await optional(path.join(OUT,'attempts.json'),[]),blocks=await optional(path.join(OUT,'blocks.json'),[]);
  return {completed:ledger.filter(e=>e.state==='completed').length,attempts:ledger.filter(e=>e.state!=='cancelled').length,
    completed_shots:blocks.filter(b=>b.state==='completed').reduce((n,b)=>n+b.shots,0),blocks:blocks.length,
    estimated_usd:ledger.reduce((n,e)=>n+(e.estimated_usd||0),0),conservative_exposure_usd:exposure(ledger),
    unresolved:ledger.filter(e=>!['completed','cancelled'].includes(e.state)).length,cap_usd:10,terminal:await optional(path.join(OUT,'complete.json'),null)};
}
export async function runBlock(){
  const p=await checkFrozen(),choice=await read(path.join(OUT,'resource_choice.json')),pool=await read(path.join(OUT,'pool.json')),pages=await read(path.join(OUT,'pages.json'));
  if(hash(await fs.readFile(path.join(OUT,'token_preview.json')))!==choice.preview_sha256)throw Error('Token bounds changed');
  const ledger=await optional(path.join(OUT,'attempts.json'),[]),blocks=await optional(path.join(OUT,'blocks.json'),[]);
  if(blocks.some(b=>b.state!=='completed')||ledger.some(e=>e.state!=='completed'))throw Error('Unresolved earlier block; stop');
  const offset=blocks.reduce((n,b)=>n+b.shots,0),n=blockSize(ledger,choice.input_bound,pool.length-offset);
  if(!n){const result={reason:offset===pool.length?'full_pool_completed':'next_complete_shot_cannot_fit_maximum_reservation',selected_n:offset,cap_usd:10};await save(path.join(OUT,'complete.json'),result);return status();}
  const selected=pool.slice(offset,offset+n),selectedOrder=selected.map((row,i)=>({row,index:offset+i})).sort((a,b)=>a.row.match_id.localeCompare(b.row.match_id)||a.index-b.index),jobs=[];
  for(const {row,index} of selectedOrder){const rotation=index%5,order=p.arms.slice(rotation).concat(p.arms.slice(0,rotation));jobs.push(...order.map(a=>job(row,a,pages,index)));}
  const lock=await fs.open(path.join(OUT,'generation.lock'),'wx');let serial=Promise.resolve(),stopped=false;
  const commit=fn=>{const next=serial.then(fn);serial=next.catch(()=>{});return next;};
  try{
    const entries=jobs.map(j=>({job:j,state:'reserved',block:blocks.length,input_bound:choice.input_bound,reserved_usd:(choice.input_bound*2.5+1024*10)/1e6,response_file:'responses/'+j.shot_id+'_'+j.arm+'.json'}));
    const block={index:blocks.length,offset,shots:n,shot_ids:selected.map(r=>r.shot_id),state:'reserved',exposure_before:exposure(ledger),maximum_reservation_usd:entries.reduce((s,e)=>s+e.reserved_usd,0),started_utc:new Date().toISOString()};
    if(block.exposure_before+block.maximum_reservation_usd>10)throw Error('Hard cap before dispatch');
    blocks.push(block);ledger.push(...entries);await save(path.join(OUT,'blocks.json'),blocks);await save(path.join(OUT,'attempts.json'),ledger);
    const token=await key();
    for(let i=0;i<entries.length&&!stopped;i+=6){
      await Promise.all(entries.slice(i,i+6).map(async entry=>{
        try{const body=await payload(entry.job,p);Object.assign(entry,{request_sha256:hash(JSON.stringify(body)),started_utc:new Date().toISOString()});await commit(()=>save(path.join(OUT,'attempts.json'),ledger));
          const raw=await api('messages',token,body);await save(path.join(OUT,entry.response_file),raw);const answer=validate(raw,p),u=raw.usage;
          const input=u.input_tokens+(u.cache_creation_input_tokens||0)+(u.cache_read_input_tokens||0);
          if(input>entry.input_bound||u.output_tokens>1024)throw Error('Reserved token bound exceeded');
          Object.assign(entry,{state:'completed',answer,usage:u,estimated_usd:cost(u),conservative_usd:cost(u,true),returned_model:raw.model,completed_utc:new Date().toISOString()});
        }catch(e){Object.assign(entry,{state:'uncertain',error:e.message,error_body:e.body,http_status:e.http_status});stopped=true;}
        await commit(()=>save(path.join(OUT,'attempts.json'),ledger));
      }));
    }
    if(stopped){for(const entry of entries)if(entry.state==='reserved')entry.state='cancelled';block.state='interrupted';await save(path.join(OUT,'attempts.json'),ledger);await save(path.join(OUT,'blocks.json'),blocks);throw Error('Stopped after unresolved request; no automatic retry');}
    block.state='completed';block.completed_utc=new Date().toISOString();block.exposure_after=exposure(ledger);await save(path.join(OUT,'blocks.json'),blocks);
    const result=await status();await save(path.join(OUT,'status.json'),result);
    if(offset+n===pool.length)await save(path.join(OUT,'complete.json'),{reason:'full_pool_completed',selected_n:pool.length,cap_usd:10});
    return status();
  }finally{await lock.close();await fs.unlink(path.join(OUT,'generation.lock'));}
}
