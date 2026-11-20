// Selective byte-range retrieval of official public-mirror GSR annotations.
// Does not access a paid API or download complete multi-gigabyte annotation files.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const OUT=path.join(ROOT,'results/soccertrack_spatial_v1');
const DATA=path.resolve(ROOT,'../soccertrack-v2');
const hash=b=>createHash('sha256').update(b).digest('hex');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
async function save(p,x){await fs.mkdir(path.dirname(p),{recursive:true});await fs.writeFile(p,JSON.stringify(x,null,2));}
async function optional(p,fallback){try{return await read(p);}catch(e){if(e.code==='ENOENT')return fallback;throw e;}}
export async function folder(id){
  const r=await fetch('https://drive.google.com/embeddedfolderview?id='+id,{signal:AbortSignal.timeout(30000)});
  if(!r.ok)throw Error('Public directory HTTP'+r.status);
  const text=await r.text(),result={};
  for(const m of text.matchAll(/<a\b[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)){
    const u=m[1].replaceAll('&amp;','&'),key=u.match(/\/(?:drive\/folders|file\/d)\/([A-Za-z0-9_-]+)/);
    if(key)result[m[2].replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim()]=key[1];
  }
  return result;
}
export async function range(file,start,length=1048576){
  start=Math.max(0,Math.floor(start));const end=Math.min(file.size-1,start+length-1);
  const base=path.join(OUT,'source/ranges',file.match+'_'+file.half+'_'+start+'_'+end);
  const audit=await optional(base+'.json',null);
  if(audit){const bytes=await fs.readFile(base+'.bin');if(hash(bytes)!==audit.sha256)throw Error('Cached source range changed');return {bytes,audit};}
  const url=new URL('https://drive.usercontent.google.com/download');
  for(const [k,v] of Object.entries({id:file.file_id,export:'download',confirm:'t',range_start:String(start)}))url.searchParams.set(k,v);
  const r=await fetch(url,{headers:{Range:'bytes='+start+'-'+end,'Accept-Encoding':'identity'},signal:AbortSignal.timeout(60000)});
  const cr=r.headers.get('content-range');
  if(r.status!==206||cr!=='bytes '+start+'-'+end+'/'+file.size){await r.body?.cancel();throw Error('Unexpected range '+r.status+' '+cr+' for '+file.match);}
  const bytes=Buffer.from(await r.arrayBuffer());if(bytes.length!==end-start+1)throw Error('Incomplete source range');
  const meta={match:file.match,half:file.half,file_id:file.file_id,start,end,size:file.size,sha256:hash(bytes),last_modified:r.headers.get('last-modified'),content_range:cr,retrieved_utc:new Date().toISOString()};
  await fs.mkdir(path.dirname(base),{recursive:true});await fs.writeFile(base+'.bin',bytes);await save(base+'.json',meta);
  return {bytes,audit:meta};
}
function endObject(text,start){
  let depth=0,quoted=false,escape=false;
  for(let i=start;i<text.length;i++){
    const c=text[i];if(quoted){if(escape)escape=false;else if(c==='\\')escape=true;else if(c==='"')quoted=false;}
    else if(c==='"')quoted=true;else if(c==='{')depth++;else if(c==='}'&&--depth===0)return i+1;
  }
  return -1;
}
export function objects(text,start=0){
  const result=[],pattern=/\{\s*"id"\s*:\s*(?:"[^"]+"|\d+)/g;let m;
  while((m=pattern.exec(text))){const end=endObject(text,m.index);if(end<0)break;
    const obj=JSON.parse(text.slice(m.index,end));pattern.lastIndex=end;
    if(obj.image_id&&obj.supercategory)result.push({object:obj,frame:Number(String(obj.image_id).slice(-6)),offset:start+m.index});
  }
  for(let i=1;i<result.length;i++)if(result[i].frame<result[i-1].frame)throw Error('Annotations are not frame ordered');
  return result;
}
export async function initialize(){
  const rootFolders=await folder('1N2Qx2qkFgRtpbHitl2Vh6sLVYGgqkWwn');
  const gsrFolders=await folder(rootFolders.gsr),rawFolders=await folder(rootFolders.raw);
  await save(path.join(OUT,'source/mirror_root.json'),rootFolders);await save(path.join(OUT,'source/raw_folders.json'),rawFolders);
  const files={};
  for(const mid of Object.keys(gsrFolders).sort()){
    const listing=await folder(gsrFolders[mid]);
    for(const half of [1,2]){
      const suffix=half===1?'1st':'2nd',name=mid+'_'+suffix+'.json';
      const pointer=await fs.readFile(path.join(DATA,'gsr',mid,name),'utf8');
      const oid=pointer.match(/oid sha256:([a-f0-9]+)/)?.[1],size=Number(pointer.match(/^size (\d+)$/m)?.[1]);
      if(!oid||!Number.isSafeInteger(size)||!listing[name])throw Error('Missing source pointer or mirror file');
      files[mid+'_'+half]={match:mid,half,file_id:listing[name],size,expected_full_sha256:oid,full_sha256_verified:false};
    }
  }
  await save(path.join(OUT,'source/files.json'),files);
  return {half_files:Object.keys(files).length,method:'Verified HTTP ranges, saved range hashes; full multi-GB file hashes are not claimed verified.'};
}
export async function prepareIndex(key){
  const files=await read(path.join(OUT,'source/files.json')),file=files[key];if(!file)throw Error('Unknown source half');
  const existing=await optional(path.join(OUT,'source/index_'+key+'.json'),null);if(existing)return existing;
  const h=await range(file,0,65536),text=h.bytes.toString('utf8');
  const info=JSON.parse(text.match(/"info"\s*:\s*(\{[\s\S]*?\})/)[1]);
  const locations=[...text.matchAll(/        \{\n            "is_labeled"/g)].map(m=>m.index);
  if(locations.length<3)throw Error('Unexpected image metadata schema');
  const first=JSON.parse(text.slice(locations[0],endObject(text,locations[0])));
  const estimate=locations[0]+(locations[1]-locations[0])*info.seq_length;
  let begin=null;
  for(let distance=0;distance<=4&&!begin;distance++){
    for(const direction of distance===0?[0]:[-1,1]){
      const sample=await range(file,Math.max(0,estimate-524288+direction*distance*1048576),1048576);
      const t=sample.bytes.toString('utf8'),i=t.indexOf('"annotations"');
      if(i>=0){const records=objects(t,sample.audit.start);if(records.length)begin={offset:records[0].offset,frame:records[0].frame};}
      if(begin)break;
    }
  }
  if(!begin)throw Error('Could not locate annotation array boundary');
  const tail=await range(file,Math.max(0,file.size-1048576),1048576),last=objects(tail.bytes.toString('utf8'),tail.audit.start);
  if(!last.length)throw Error('Missing final annotation records');
  const final=last[last.length-1];if(final.frame!==info.seq_length)throw Error('Mirror frame count mismatch');
  const index={...file,info,image_metadata:first,anchors:[begin,{frame:final.frame,offset:final.offset}],header_range:h.audit,notes:'Frame suffix is1-based; decoded video frame index is suffix-1. Validate released metadata dimensions against actual saved image geometry.'};
  await save(path.join(OUT,'source/index_'+key+'.json'),index);return index;
}
export async function retrieveHalf(key,shots){
  const index=await prepareIndex(key),targets=[...new Set(shots.flatMap(s=>[s.frame_index+1,s.event_frame]))].sort((a,b)=>a-b);
  const needed=new Set(targets);let hits=0;
  for(const target of targets){
    if(await optional(path.join(OUT,'snapshots',key+'_'+target+'.json'),null)){needed.delete(target);hits++;}
  }
  for(const target of targets){
    if(!needed.has(target))continue;
    let found=false;
    for(let iteration=0;iteration<16&&!found;iteration++){
      const anchors=index.anchors.sort((a,b)=>a.frame-b.frame);
      const lower=anchors.filter(a=>a.frame<target).at(-1)||{frame:0,offset:index.anchors[0].offset};
      const upper=anchors.find(a=>a.frame>target)||{frame:index.info.seq_length+1,offset:index.size};
      const predicted=lower.offset+(target-lower.frame)/(upper.frame-lower.frame)*(upper.offset-lower.offset);
      const length=iteration<6?1048576:2097152;
      const sample=await range(index,Math.max(0,Math.min(index.size-length,Math.round(predicted-length/2))),length);
      const records=objects(sample.bytes.toString('utf8'),sample.audit.start);
      if(!records.length)throw Error('No annotation records at interpolated position');
      const distinct=[...new Set(records.map(r=>r.frame))];
      for(const frame of distinct.slice(1,-1)){
        const group=records.filter(r=>r.frame===frame),anchor={frame,offset:group[0].offset};
        if(!index.anchors.some(a=>a.frame===frame))index.anchors.push(anchor);
        if(needed.has(frame)){
          await save(path.join(OUT,'snapshots',key+'_'+frame+'.json'),{match:index.match,half:index.half,annotation_frame:frame,video_frame_index:frame-1,image_id:group[0].object.image_id,objects:group.map(r=>r.object),source_range:sample.audit,file_id:index.file_id,expected_full_sha256:index.expected_full_sha256,full_sha256_verified:false});
          needed.delete(frame);hits++;
        }
      }
      if(!index.anchors.some(a=>a.frame===records[0].frame))index.anchors.push({frame:records[0].frame,offset:records[0].offset});
      if(!index.anchors.some(a=>a.frame===records.at(-1).frame))index.anchors.push({frame:records.at(-1).frame,offset:records.at(-1).offset});
      await save(path.join(OUT,'source/index_'+key+'.json'),index);
      found=!needed.has(target);
    }
    if(!found)throw Error('Could not retrieve exact requested frame '+target+' in '+key);
  }
  if(needed.size)throw Error('Incomplete source snapshots');
  return {half:key,shots:shots.length,frames:hits,source_bytes_cached:(await fs.readdir(path.join(OUT,'source/ranges'))).filter(n=>n.endsWith('.bin')).length};
}
export async function retrieveMatches(matchIDs){
  const rows=await read(path.join(ROOT,'results/soccertrack_extensive_v1/frames.json')),result=[];
  for(const mid of matchIDs)for(const half of [1,2])result.push(await retrieveHalf(mid+'_'+half,rows.filter(r=>r.match_id===mid&&r.half===half)));
  return result;
}
export async function calibration(matchIDs){
  const folders=await read(path.join(OUT,'source/raw_folders.json')),result=[];
  for(const mid of matchIDs){
    const listing=await folder(folders[mid]);
    for(const name of ['homography.npy','camera_intrinsics.npz','mapx.npy','mapy.npy']){
      const filename=mid+'_'+name,pointer=await fs.readFile(path.join(DATA,'raw',mid,filename),'utf8');
      const oid=pointer.match(/oid sha256:([a-f0-9]+)/)?.[1],size=Number(pointer.match(/^size (\d+)$/m)?.[1]);
      if(!oid||!size||!listing[filename])throw Error('Calibration resource missing');
      const full=size<=65536,length=full?size:256;
      const file={match:mid+'_'+name,half:0,file_id:listing[filename],size};
      const sample=await range(file,0,length);
      if(full&&hash(sample.bytes)!==oid)throw Error('Canonical calibration SHA mismatch');
      const output=path.join(OUT,'source/calibration',filename+(full?'':'.header'));
      await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,sample.bytes);
      result.push({match:mid,name,path:path.relative(ROOT,output).replaceAll('\\','/'),expected_full_sha256:oid,full_sha256_verified:full,source_range:sample.audit});
    }
  }
  await save(path.join(OUT,'source/calibration_'+matchIDs.join('_')+'.json'),result);return result.map(x=>({match:x.match,name:x.name,full_sha256_verified:x.full_sha256_verified}));
}
