// Resource-triggered provider fallback. Importing never makes paid calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import * as original from './run.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const PARENT=path.join(ROOT,'results/soccertrack_icl_v1');
const OUT=path.join(ROOT,'results/soccertrack_icl_sonnet_v1');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const hash=b=>createHash('sha256').update(b).digest('hex');
const sha=async p=>hash(await fs.readFile(p));
async function save(p,value){await fs.mkdir(path.dirname(p),{recursive:true});await fs.writeFile(p+'.tmp',JSON.stringify(value,null,2));await fs.rename(p+'.tmp',p);}
async function optional(p,fallback){try{return await read(p);}catch(e){if(e.code==='ENOENT')return fallback;throw e;}}
async function credential(){const s=await fs.readFile(path.join(ROOT,'../keys.txt'),'utf8');const values=[...new Set(s.match(/sk-ant-[A-Za-z0-9_-]+/g)||[])];if(values.length!==1)throw Error('Unique provider credential required');return values[0];}
async function api(endpoint,key,body){
  const r=await fetch('https://api.anthropic.com/v1/'+endpoint,{method:'POST',headers:{'x-api-key':key,'anthropic-version':'2023-06-01','Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(150000)});
  const raw=await r.json();
  if(!r.ok){const e=Error('Anthropic HTTP '+r.status+'; '+(raw.error?.type||'unknown')+'; no automatic retry');e.http_status=r.status;e.body=raw;e.request_id=r.headers.get('request-id')||r.headers.get('x-request-id');throw e;}
  return raw;
}
export function cost(u,conservative=false){
  const ni=u.input_tokens,no=u.output_tokens,nw=u.cache_creation_input_tokens||0,nr=u.cache_read_input_tokens||0;
  if(![ni,no,nw,nr].every(Number.isInteger)||[ni,no,nw,nr].some(x=>x<0))throw Error('Invalid provider usage');
  return (conservative?(ni+nw+nr)*2.5+no*10:ni*2+nw*2.5+nr*.1+no*10)/1e6;
}
export function exposure(ledger){return ledger.reduce((s,e)=>s+(e.state==='completed'?e.conservative_usd:e.reserved_usd),0);}
export function canReserve(ledger,entry,prior){
  if(ledger.some(e=>e.state==='uncertain'))throw Error('Uncertain new request; stop dispatch');
  if(!Number.isFinite(prior)||prior<0||!Number.isFinite(entry.reserved_usd)||entry.reserved_usd<0)throw Error('Invalid reservation');
  return ledger.length<480&&prior+exposure(ledger)+entry.reserved_usd<=10;
}
export async function payload(job,p,pages){
  const openai=await original.payload(job,p,pages);
  const content=openai.input[0].content.map(part=>{
    if(part.type==='input_text')return {type:'text',text:part.text};
    const match=part.image_url.match(/^data:([^;]+);base64,(.*)$/s);
    if(!match)throw Error('Unexpected image data URL');
    return {type:'image',source:{type:'base64',media_type:match[1],data:match[2]}};
  });
  // Range is checked after parsing; shape is enforced by constrained JSON decoding.
  const schema={type:'object',properties:{xg:{type:'number'},reason:{type:'string'}},required:['xg','reason'],additionalProperties:false};
  return {model:'claude-sonnet-5-5',max_tokens:1024,inference_geo:'global',output_config:{effort:'low',format:{type:'json_schema',schema}},messages:[{role:'user',content}]};
}
export function validate(raw){
  if(!(raw.model==='claude-sonnet-5-5'||raw.model?.startsWith('claude-sonnet-5-5-'))||raw.type!=='message'||raw.stop_reason!=='end_turn')throw Error('Unexpected/incomplete provider response');
  const text=raw.content.filter(x=>x.type==='text').map(x=>x.text).join('');const answer=JSON.parse(text);
  if(Object.keys(answer).sort().join(',')!=='reason,xg'||typeof answer.reason!=='string'||typeof answer.xg!=='number'||!Number.isFinite(answer.xg)||answer.xg<0||answer.xg>1)throw Error('Invalid raw probability');
  cost(raw.usage);return answer;
}
export async function setup(){
  const base=await original.checkFrozen(),decl=await read(path.join(OUT,'declaration.json')),pages=await read(path.join(PARENT,'pages.json')),jobs=await read(path.join(PARENT,'preview/jobs.json')),key=await credential();
  if(await optional(path.join(OUT,'protocol.json'),null))throw Error('Fallback protocol already exists');
  if(await sha(path.join(PARENT,'attempts.json'))!==decl.parent_ledger_sha256)throw Error('Parent ledger changed');
  const counts=await optional(path.join(OUT,'token_counts.json'),{});
  for(let i=0;i<jobs.length;i+=4){
    await Promise.all(jobs.slice(i,i+4).map(async job=>{
      if(counts[job.match_id+':'+job.arm])return;
      const request=await payload(job,base,pages);const n=await api('messages/count_tokens',key,{model:request.model,messages:request.messages});
      if(!Number.isInteger(n.input_tokens)||n.input_tokens<=0)throw Error('No provider token count');
      counts[job.match_id+':'+job.arm]={input_tokens:n.input_tokens,reserved_usd:((n.input_tokens+1024)*2.5+1024*10)/1e6};
    }));
    await save(path.join(OUT,'token_counts.json'),counts);
  }
  const pool=await read(path.join(PARENT,'evaluation/selection.json'));
  const reservation=Math.max(...Object.values(counts).map(v=>v.reserved_usd));
  const forecasts={};
  for(const n of [80,60]){
    const expected=pool.slice(0,n).reduce((sum,row)=>sum+base.arms.reduce((s,a)=>s+((counts[row.match_id+':'+a].input_tokens+256)*2.5+128*10)/1e6,0),0);
    forecasts[n]=decl.prior_conservative_exposure_usd+1.15*expected+4*reservation+.20;
  }
  const n=[80,60].find(n=>forecasts[n]<=10);
  if(!n)throw Error('Complete fallback study cannot fit original combined $10 cap; no paid calls');
  const selected=pool.slice(0,n),forecast=forecasts[n];
  if(new Set(selected.map(r=>r.match_id)).size!==10)throw Error('Fallback cohort does not cover all10 matches');
  const fallbackJobs=(await read(path.join(PARENT,'evaluation/jobs.json'))).slice(0,n*base.arms.length);
  await save(path.join(OUT,'selection.json'),selected);await save(path.join(OUT,'jobs.json'),fallbackJobs);
  await save(path.join(OUT,'resource_amendment.json'),{reason:'Provider token counts plus retained original7 rejection reservations require a smaller natural prefix. This choice precedes all fallback generation and accuracy inspection.',candidates:[80,60],forecasts_usd:forecasts,selected_shots:n,unchanged:'All6 conditions,examples,prompts and match exclusions are unchanged;all10 matches remain represented.'});
  const protocol={...base,model:'claude-sonnet-5-5',provider:'anthropic',prior_conservative_exposure_usd:decl.prior_conservative_exposure_usd,prior_estimated_usd:decl.prior_estimated_usd,
    parent_ledger_sha256:decl.parent_ledger_sha256,parent_protocol_sha256:decl.parent_protocol_sha256,recovery_probe_sha256:decl.recovery_probe_sha256,
    declaration_sha256:await sha(path.join(OUT,'declaration.json')),runner_sha256:await sha(fileURLToPath(import.meta.url)),
    jobs_sha256:await sha(path.join(PARENT,'evaluation/jobs.json')),selection_sha256:await sha(path.join(PARENT,'evaluation/selection.json')),
    fallback_jobs_sha256:await sha(path.join(OUT,'jobs.json')),fallback_selection_sha256:await sha(path.join(OUT,'selection.json')),resource_amendment_sha256:await sha(path.join(OUT,'resource_amendment.json')),planned_calls:fallbackJobs.length,
    expected_combined_conservative_usd:forecast,workers:4,price_source:'https://platform.claude.com/docs/en/about-claude/pricing',
    amendment:'Alternative provider selected solely for credit exhaustion, before accuracy inspection. Fresh complete6-arm study on the usage-selected prefix. Original Sol partial results are never pooled. Original7 rejection reservations remain counted. Generic text and image bytes match the parent; provider-native JSON output format and low effort replace Responses API settings.'};
  if(forecast>10)throw Error('Complete fallback study cannot fit original combined $10 cap; no paid calls');
  await save(path.join(OUT,'protocol.json'),protocol);return {conditions_counted:Object.keys(counts).length,forecast_combined_usd:forecast,shots:selected.length,planned_calls:selected.length*base.arms.length};
}
export async function checkFrozen(){
  await original.checkFrozen();const p=await read(path.join(OUT,'protocol.json'));
  for(const [field,filename] of [['parent_ledger_sha256','attempts.json'],['parent_protocol_sha256','protocol.json'],['recovery_probe_sha256','recovery_probe.json'],['jobs_sha256','evaluation/jobs.json'],['selection_sha256','evaluation/selection.json']])
    if(await sha(path.join(PARENT,filename))!==p[field])throw Error('Parent artifact changed '+filename);
  if(await sha(fileURLToPath(import.meta.url))!==p.runner_sha256||await sha(path.join(OUT,'declaration.json'))!==p.declaration_sha256)throw Error('Fallback code/declaration changed');
  for(const [field,name] of [['fallback_jobs_sha256','jobs.json'],['fallback_selection_sha256','selection.json'],['resource_amendment_sha256','resource_amendment.json']])
    if(await sha(path.join(OUT,name))!==p[field])throw Error('Fallback frozen data changed');
  return p;
}
export async function status(){
  const p=await read(path.join(OUT,'protocol.json')),ledger=await optional(path.join(OUT,'attempts.json'),[]);
  return {completed:ledger.filter(e=>e.state==='completed').length,attempts:ledger.length,planned:p.planned_calls,uncertain:ledger.filter(e=>e.state!=='completed').length,
    fallback_estimated_usd:ledger.reduce((s,e)=>s+(e.estimated_usd||0),0),combined_estimated_usd:p.prior_estimated_usd+ledger.reduce((s,e)=>s+(e.estimated_usd||0),0),combined_conservative_exposure_usd:p.prior_conservative_exposure_usd+exposure(ledger),cap_usd:10};
}
export async function runBatch(limit=24){
  const p=await checkFrozen(),pages=await read(path.join(PARENT,'pages.json')),jobs=await read(path.join(OUT,'jobs.json')),key=await credential(),ledger=await optional(path.join(OUT,'attempts.json'),[]);
  if(ledger.some(e=>e.state!=='completed'))throw Error('Unresolved provider request; no retry');
  if(JSON.stringify(ledger.map(e=>e.job))!==JSON.stringify(jobs.slice(0,ledger.length)))throw Error('Schedule mismatch');
  const selected=jobs.slice(ledger.length,ledger.length+limit),lock=await fs.open(path.join(OUT,'generation.lock'),'wx');let serial=Promise.resolve(),stopped=false;
  const transaction=fn=>{const result=serial.then(fn);serial=result.catch(()=>{});return result;};
  try{
    for(let i=0;i<selected.length&&!stopped;i+=4){
      const prepared=await Promise.all(selected.slice(i,i+4).map(async job=>{
        const request=await payload(job,p,pages),n=await api('messages/count_tokens',key,{model:request.model,messages:request.messages});
        if(!Number.isInteger(n.input_tokens)||n.input_tokens<=0||n.input_tokens>60000)throw Error('Count unavailable');
        const bound=n.input_tokens+1024;
        return {request,entry:{job,state:'reserved',input_bound:bound,reserved_usd:(bound*2.5+1024*10)/1e6,request_sha256:hash(JSON.stringify(request)),response_file:'responses/'+job.shot_id+'_'+job.arm+'.json',started_utc:new Date().toISOString()}};
      }));
      const temporary=ledger.slice();for(const {entry} of prepared){if(!canReserve(temporary,entry,p.prior_conservative_exposure_usd))throw Error('Combined $10 hard cap reached');temporary.push(entry);}
      ledger.push(...prepared.map(x=>x.entry));await save(path.join(OUT,'attempts.json'),ledger);
      await Promise.all(prepared.map(async({request,entry})=>{
        try{
          const raw=await api('messages',key,request);await save(path.join(OUT,entry.response_file),raw);const answer=validate(raw);
          const ni=raw.usage.input_tokens+(raw.usage.cache_creation_input_tokens||0)+(raw.usage.cache_read_input_tokens||0);
          if(ni>entry.input_bound||raw.usage.output_tokens>1024)throw Error('Reserved token count exceeded');
          Object.assign(entry,{state:'completed',usage:raw.usage,answer,returned_model:raw.model,estimated_usd:cost(raw.usage),conservative_usd:cost(raw.usage,true),completed_utc:new Date().toISOString()});
        }catch(e){Object.assign(entry,{state:'uncertain',error:e.message,http_status:e.http_status,error_body:e.body,request_id:e.request_id});stopped=true;}
        await transaction(()=>save(path.join(OUT,'attempts.json'),ledger));
      }));
      await save(path.join(OUT,'status.json'),await status());
    }
    const result=await status();if(result.completed===jobs.length)await save(path.join(OUT,'complete.json'),result);
    if(stopped)throw Error('Provider stopped; inspect saved error. No automatic retry.');return result;
  }finally{await lock.close();await fs.unlink(path.join(OUT,'generation.lock'));}
}
