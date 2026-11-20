// Offline checks of financial bounds, including a deliberately uncached run.
import assert from 'node:assert/strict';
import {blockSize,cost,exposure} from './run.mjs';
assert.equal(cost({input_tokens:100,output_tokens:20,cache_creation_input_tokens:200,cache_read_input_tokens:1000}),.001);
assert.equal(cost({input_tokens:100,output_tokens:20,cache_creation_input_tokens:200,cache_read_input_tokens:1000},true),.001075);
assert.equal(blockSize([{state:'completed',conservative_usd:9.99}],9000,304),0);
assert.equal(blockSize([],9000,304),10);
assert.throws(()=>blockSize([{state:'uncertain',reserved_usd:1}],9000,304));
assert.throws(()=>cost({input_tokens:1,output_tokens:1,cache_creation:{ephemeral_1h_input_tokens:1}}));
assert.throws(()=>cost({input_tokens:-1,output_tokens:1}));
for(const cached of [false,true]){
  const ledger=[];let shots=0;
  while(shots<304){
    const n=blockSize(ledger,10000,304-shots);if(!n)break;
    const entries=Array.from({length:n*5},()=>({state:'reserved',reserved_usd:(10000*2.5+1024*10)/1e6}));
    ledger.push(...entries);assert.ok(exposure(ledger)<=10);
    for(const entry of entries){entry.state='completed';entry.conservative_usd=cost(cached?{input_tokens:1000,cache_read_input_tokens:9000,output_tokens:1024}:{input_tokens:10000,output_tokens:1024},true);}
    shots+=n;assert.equal(ledger.length,shots*5);assert.ok(exposure(ledger)<=10);
  }
  assert.ok(shots>0);assert.equal(ledger.length%5,0);assert.ok(shots===304||blockSize(ledger,10000,304-shots)===0);
}
console.log('Passed7 cost/guard checks and2 full-budget simulations, including no cache hits.');
