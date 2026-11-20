// Offline budget checks; no network calls and no credentials.
import assert from 'node:assert/strict';
import {cost,canReserve,exposure} from './run.mjs';
assert.equal(cost({input_tokens:1000,output_tokens:100}),.003);
assert.equal(cost({input_tokens:1000,output_tokens:100,input_tokens_details:{cached_tokens:1000}}),.0011);
assert.equal(cost({input_tokens:1000,output_tokens:100},true),.0035);
assert.equal(canReserve([{state:'reserved',reserved_usd:9.9}],{reserved_usd:.11}),false);
assert.equal(canReserve([{state:'completed',conservative_usd:9.8}],{reserved_usd:.1}),true);
assert.equal(exposure([{state:'completed',conservative_usd:.2},{state:'reserved',reserved_usd:.3}]),.5);
assert.throws(()=>canReserve([{state:'uncertain',reserved_usd:.1}],{reserved_usd:.1}));
assert.throws(()=>cost({input_tokens:1,output_tokens:-1}));
assert.throws(()=>cost({input_tokens:1,output_tokens:0,input_tokens_details:{cached_tokens:2}}));
assert.throws(()=>canReserve([],{reserved_usd:NaN}));
assert.equal(canReserve(Array.from({length:800},()=>({state:'completed',conservative_usd:0})),{reserved_usd:0}),false);
console.log('11 offline budget checks passed');
