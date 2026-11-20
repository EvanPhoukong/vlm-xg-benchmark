"""No automatic retries; globally budgeted sequential inference in batches of ten."""
import argparse, os, time
from pathlib import Path
import common as c

def run(stage):
    c.campaign_protocol(); folder=c.OUT/stage; protocol=c.read(folder/'protocol.json')
    if not protocol: raise ValueError('Stage has not been frozen')
    if c.sha(Path(__file__))!=protocol['runner_sha256'] or c.sha(Path(__file__).with_name('common.py'))!=protocol['engine_sha256']: raise ValueError('Frozen inference code changed')
    ledger=c.read(c.OUT/'attempts.json',[]); done=[a for a in ledger if a['stage']==stage]
    if [{k:a[k] for k in ('shot_id','condition','effort','image','image_sha256')} for a in done]!=protocol['schedule'][:len(done)]: raise ValueError('Saved schedule mismatch')
    counts=c.read(c.OUT/'token_counts.json',{}); key=c.prior.credential()
    lock=c.OUT/'generation.lock'; fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    try:
        for index,job in enumerate(protocol['schedule'][len(done):],len(done)+1):
            path=c.OUT/job['image']
            if c.sha(path)!=job['image_sha256']: raise ValueError('Frozen image changed')
            payload=c.request_body(protocol['prompt'],path.read_bytes(),job['effort']); countkey=job['image_sha256']
            if countkey not in counts:
                counted=c.prior.post('responses/input_tokens',{k:payload[k] for k in ('model','input','text')},key)
                n=counted.get('input_tokens')
                if not isinstance(n,int) or n<=0: raise ValueError('Input count unavailable; no generation attempted')
                counts[countkey]=n; c.save(c.OUT/'token_counts.json',counts)
            bound=counts[countkey]+1024; c.check_budget(ledger,stage,bound)
            attempt={**job,'stage':stage,'state':'started','input_bound':bound,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            ledger.append(attempt); c.save(c.OUT/'attempts.json',ledger); start=time.monotonic()
            try:
                response=c.prior.post('responses',payload,key)
                name=f'{stage}/response_{index:04d}.json'; c.save(c.OUT/name,response)
                attempt.update(response_file=name,usage=response.get('usage') or {},seconds=round(time.monotonic()-start,3))
                answer=c.prior.validate(response)
                if response.get('reasoning',{}).get('effort')!=job['effort']: raise ValueError('Returned reasoning differs')
                if attempt['usage']['input_tokens']>bound or attempt['usage']['output_tokens']>c.MAX_OUTPUT: raise ValueError('Reserved token bound exceeded')
                attempt.update(state='completed',answer=answer,model=response['model'])
            except Exception as e:
                attempt.update(state='failed_or_uncertain',error=str(e)); c.save(c.OUT/'attempts.json',ledger)
                c.save(c.OUT/'status.json',{'status':'stopped','stage':stage,'reason':str(e),'usage':c.usage(ledger),'prior_pilot_usd':c.PRIOR_COST})
                raise
            c.save(c.OUT/'attempts.json',ledger)
            c.save(c.OUT/'status.json',{'status':'running','stage':stage,'completed':index,'planned':len(protocol['schedule']),'usage':c.usage(ledger),'prior_pilot_usd':c.PRIOR_COST})
            if index%10==0 or index==len(protocol['schedule']): print(f"{stage}: {index}/{len(protocol['schedule'])}; campaign estimate ${c.usage(ledger)['estimated_usd']+c.PRIOR_COST:.5f}",flush=True)
        c.save(folder/'complete.json',{'responses':len(protocol['schedule']),'usage':c.usage([a for a in ledger if a['stage']==stage])})
        print('Stage complete; no automatic new stage or cohort expansion.',flush=True)
    finally: lock.unlink()
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['reasoning','rendering','confirmation','invariance']); run(ap.parse_args().stage)
