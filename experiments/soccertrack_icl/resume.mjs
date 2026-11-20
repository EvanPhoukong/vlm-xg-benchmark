// Importing performs no network requests or paid work.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const OUT=path.join(ROOT,'results/soccertrack_icl_resume_v1');
const PARENT=path.join(ROOT,'results/soccertrack_icl_v1');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const hash=b=>createHash('sha256').update(b).digest('hex');
const fileHash=async p=>hash(await fs.readFile(p));
async function save(p,value){await fs.mkdir(path.dirname(p),{recursive:true});await fs.writeFile(p+'.tmp',JSON.stringify(value,null,2));await fs.rename(p+'.tmp',p);}
async function existing(p,fallback){try{return await read(p);}catch(e){if(e.code==='ENOENT')return fallback;throw e;}}
async function credential(){
  const text=await fs.readFile(path.join(ROOT,'../keys.txt'),'utf8');
  const matches=[...new Set(text.match(/sk-(?!ant-)[A-Za-z0-9_-]+/g)||[])];
  if(matches.length!==1)throw Error('One OpenAI credential required; never displayed');
  return matches[0];
}
async function api(endpoint,key,payload){
  const r=await fetch('https://api.openai.com/v1/'+endpoint,{method:'POST',headers:{Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(150000)});
  if(!r.ok)throw Error('API HTTP '+r.status+'; no retry');
  return r.json();
}
export function cost(u,conservative=false){
  const ni=u.input_tokens,no=u.output_tokens,nc=u.input_tokens_details?.cached_tokens||0;
  if(![ni,no,nc].every(Number.isInteger)||ni<0||no<0||nc<0||nc>ni)throw Error('Invalid token usage');
  return (conservative?ni*2*1.25+no*10:(ni-nc)*2+nc*.1+no*10)/1e6;
}
export function exposure(ledger){return ledger.reduce((s,e)=>s+(e.state==='completed'?e.conservative_usd:e.reserved_usd),0);}
export function canReserve(ledger,entry,prior=0){
  if(ledger.some(e=>e.state==='uncertain'))throw Error('Uncertain charge; stop dispatch');
  if(!Number.isFinite(entry.reserved_usd)||entry.reserved_usd<0)throw Error('Invalid reservation');
  return ledger.length<2000&&prior+exposure(ledger)+entry.reserved_usd<=10;
}
async function imagePart(relative,expected){
  const filename=path.resolve(ROOT,relative);
  if(!filename.startsWith(ROOT+path.sep))throw Error('Image path escapes repository');
  const bytes=await fs.readFile(filename);
  if(hash(bytes)!==expected)throw Error('Image changed: '+relative);
  return {type:'input_image',detail:'high',image_url:'data:image/'+(relative.endsWith('.jpg')?'jpeg':'png')+';base64,'+bytes.toString('base64')};
}
export async function payload(job,p,pages){
  const parts=[];
  if(job.arm!=='zero'){
    parts.push({type:'input_text',text:job.arm.includes('teacher')?p.teacher_intro:p.outcome_intro});
    for(const page of pages[job.match_id][job.arm])parts.push(await imagePart(page.path,page.sha256));
    parts.push({type:'input_text',text:'Historical examples END. The following separate image is the NEW QUERY; its outcome is unknown.'});
  }
  parts.push(await imagePart(job.image,job.sha256));parts.push({type:'input_text',text:p.prompt});
  return {model:p.model,store:false,service_tier:'default',reasoning:{effort:'low'},max_output_tokens:1024,
    input:[{role:'user',content:parts}],text:{format:{type:'json_schema',name:'shot_prediction',strict:true,
      schema:{type:'object',properties:{xg:{type:'number',minimum:0,maximum:1},reason:{type:'string'}},required:['xg','reason'],additionalProperties:false}}}};
}
function validate(raw,p){
  if(raw.status!=='completed'||!(raw.model===p.model||raw.model?.startsWith(p.model+'-'))||raw.reasoning?.effort!=='low')throw Error('Unexpected model/settings or incomplete output');
  const text=raw.output.filter(x=>x.type==='message').flatMap(x=>x.content.filter(p=>p.type==='output_text').map(p=>p.text)).join('');
  const answer=JSON.parse(text);
  if(Object.keys(answer).sort().join(',')!=='reason,xg'||typeof answer.reason!=='string'||typeof answer.xg!=='number'||!Number.isFinite(answer.xg)||answer.xg<0||answer.xg>1)throw Error('Invalid raw probability');
  cost(raw.usage);return answer;
}
export async function checkFrozen(){
  const p=await read(path.join(OUT,'protocol.json'));
  for(const [field,relative] of [['pages_sha256','pages.json'],['examples_sha256','examples.json'],['pool_sha256','pool.json'],['baselines_sha256','baselines.json'],['preview_sha256','preview/jobs.json']])
    if(await fileHash(path.join(PARENT,relative))!==p[field])throw Error('Frozen artifact changed '+relative);
  if(await fileHash(fileURLToPath(import.meta.url))!==p.runner_sha256)throw Error('Runner changed after freeze');
  if(await fileHash(path.join(ROOT,'experiments/soccertrack_icl/experiment.py'))!==p.python_sha256)throw Error('Python changed after freeze');
  for(const [field,filename] of [['parent_ledger_sha256',p.parent_ledger_path],['original_protocol_sha256',p.original_protocol_path],['recovery_probe_sha256',p.recovery_probe_path],['sonnet_ledger_sha256',p.sonnet_ledger_path]])
    if(await fileHash(path.join(ROOT,filename))!==p[field])throw Error('Frozen prior accounting changed');
  if(await fileHash(path.join(OUT,'evaluation/jobs.json'))!==p.resume_jobs_sha256)throw Error('Resume schedule changed');
  return p;
}
export async function status(){
  const ledger=await existing(path.join(OUT,'attempts.json'),[]);
  const p=await read(path.join(OUT,'protocol.json'));
  return {combined_estimated_usd:p.prior_estimated_usd+ledger.reduce((s,e)=>s+(e.estimated_usd||0),0),combined_conservative_exposure_usd:p.prior_conservative_exposure_usd+exposure(ledger),total_sol_completed:96+ledger.filter(e=>e.state==='completed').length,attempts:ledger.length,completed:ledger.filter(e=>e.state==='completed').length,estimated_usd:ledger.reduce((s,e)=>s+(e.estimated_usd||0),0),conservative_exposure_usd:exposure(ledger),cap_usd:10,uncertain:ledger.filter(e=>e.state!=='completed').length,
    stages:Object.fromEntries(['evaluation'].map(s=>[s,ledger.filter(e=>e.stage===s&&e.state==='completed').length]))};
}
export async function countPreview(){
  const p=await checkFrozen(),pages=await read(path.join(PARENT,'pages.json')),jobs=await read(path.join(PARENT,'preview/jobs.json')),key=await credential(),result={};
  for(let i=0;i<jobs.length;i+=6){
    await Promise.all(jobs.slice(i,i+6).map(async job=>{
      const request=await payload(job,p,pages);
      const r=await api('responses/input_tokens',key,{model:request.model,input:request.input,text:request.text});
      if(!Number.isInteger(r.input_tokens)||r.input_tokens<=0)throw Error('No counted tokens');
      result[job.match_id+':'+job.arm]={input_tokens:r.input_tokens,max_reserved_usd:cost({input_tokens:r.input_tokens+1024,output_tokens:1024},true)};
    }));
    await save(path.join(OUT,'token_preview.json'),result);
  }
  return {counted_conditions:Object.keys(result).length,min_tokens:Math.min(...Object.values(result).map(x=>x.input_tokens)),max_tokens:Math.max(...Object.values(result).map(x=>x.input_tokens))};
}
export async function runBatch(stage,limit=30){
  if(stage!=='evaluation')throw Error('Invalid stage');
  const p=await checkFrozen(),pages=await read(path.join(PARENT,'pages.json'));
  const jobs=await read(path.join(OUT,stage,'jobs.json')),key=await credential(),ledger=await existing(path.join(OUT,'attempts.json'),[]);
  if(ledger.some(e=>e.state!=='completed'))throw Error('Prior unresolved request; no dispatch');
  const done=ledger.filter(e=>e.stage===stage);
  if(JSON.stringify(done.map(e=>e.job))!==JSON.stringify(jobs.slice(0,done.length)))throw Error('Schedule mismatch');
  const selected=jobs.slice(done.length,done.length+limit),lock=await fs.open(path.join(OUT,'generation.lock'),'wx');
  let serial=Promise.resolve(),stopped=false;
  function transaction(fn){const next=serial.then(fn);serial=next.catch(()=>{});return next;}
  try{
    for(let offset=0;offset<selected.length&&!stopped;offset+=6){
      const prepared=await Promise.all(selected.slice(offset,offset+6).map(async job=>{
        const request=await payload(job,p,pages),counted=await api('responses/input_tokens',key,{model:request.model,input:request.input,text:request.text});
        if(!Number.isInteger(counted.input_tokens)||counted.input_tokens<=0||counted.input_tokens>60000)throw Error('Input count unavailable/out of bounds');
        const bound=counted.input_tokens+1024;
        const entry={stage,job,state:'reserved',request_sha256:hash(JSON.stringify(request)),input_bound:bound,reserved_usd:cost({input_tokens:bound,output_tokens:1024},true),started_utc:new Date().toISOString(),response_file:stage+'/responses/'+job.shot_id+'_'+job.arm+'.json'};
        return {entry,request};
      }));
      let temporary=[...ledger];for(const {entry} of prepared){if(!canReserve(temporary,entry,p.prior_conservative_exposure_usd))throw Error('Hard $10 cap reached before wave');temporary.push(entry);}
      ledger.push(...prepared.map(x=>x.entry));await save(path.join(OUT,'attempts.json'),ledger);
      await Promise.all(prepared.map(async({entry,request})=>{
        try{
          const raw=await api('responses',key,request);await save(path.join(OUT,entry.response_file),raw);
          const answer=validate(raw,p);
          if(raw.usage.input_tokens>entry.input_bound||raw.usage.output_tokens>1024)throw Error('Token reservation exceeded');
          Object.assign(entry,{state:'completed',usage:raw.usage,answer,returned_model:raw.model,estimated_usd:cost(raw.usage),conservative_usd:cost(raw.usage,true),completed_utc:new Date().toISOString()});
        }catch(e){entry.state='uncertain';entry.error=String(e.message);stopped=true;}
        await transaction(async()=>{await save(path.join(OUT,'attempts.json'),ledger);});
      }));
      await save(path.join(OUT,'status.json'),await status());
    }
    const result=await status();
    if(ledger.filter(e=>e.stage===stage&&e.state==='completed').length===jobs.length)await save(path.join(OUT,stage,'complete.json'),result);
    if(stopped)throw Error('Generation stopped; inspect uncertain ledger and saved raw responses');
    return result;
  }finally{await lock.close();await fs.unlink(path.join(OUT,'generation.lock'));}
}
