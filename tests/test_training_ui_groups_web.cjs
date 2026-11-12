// Run with: node --test tests/test_training_ui_groups_web.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.dataset={}; this.value=''; this.classList={add(){},remove(){}}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  addEventListener() {}
  dispatchEvent() {}
  querySelector(selector) { const value=selector.match(/^\[value=(.*)\]$/)?.[1]; return this.children.find(el=>el.value===value); }
  querySelectorAll(tag) { return this.children.flatMap(el=>[...((el.tag===tag||(tag==='input:checked'&&el.tag==='input'&&el.checked))?[el]:[]),...el.querySelectorAll(tag)]); }
}
function setup() {
  const elements=new Map(['groups','groups-result','sampling-mode','dataset-candidates','apply-datasets','wandb-mode','auto-norm','allow-busy'].map(id=>[id,new Element('div')]));
  elements.get('sampling-mode').value='weighted_replacement';
  const context=vm.createContext({URLSearchParams,Event:class {},location:{hash:''},document:{
    getElementById:id=>elements.get(id), createElement:tag=>new Element(tag), querySelectorAll:()=>[],
  }});
  const source=fs.readFileSync(path.join(__dirname,'../scripts/training_ui_web/app.js'),'utf8');
  // Load the actual form logic; startup/polling bindings are exercised by the browser.
  vm.runInContext(source.slice(0,source.indexOf("$('preview').addEventListener")),context);
  vm.runInContext(`baseline='example'; fields.set('dataset_root',{input:{value:'/data'}});
    updatePath=()=>{}; notice=()=>{};
    addGroup({name:'A',weight:7,episodes:[99]}); addGroup({name:'B',weight:3,episodes:[]});`,context);
  return {context,elements,run:code=>vm.runInContext(code,context)};
}
const detected={episodes:4,frames:40,groups:[
  {name:'A',episodes:[0,1],frames:20,source_root:'/old/A-subtask'},
  {name:'B',episodes:[4],frames:10,source_root:'/old/B-subtask'},
  {name:'C',episodes:[8],frames:10,source_root:'/old/C'},
]};

test('auto-fill keeps weights, defaults new groups, and compacts episode ranges',async()=>{
  const {context,run,elements}=setup();
  context.result=detected;
  run('api=async()=>result');
  await run('detectGroups()');
  assert.deepEqual(JSON.parse(run('JSON.stringify(groupInputs())')), [
    {name:'A',weight:7,episodes:'0-1'}, {name:'B',weight:3,episodes:'4'}, {name:'C',weight:1,episodes:'8'},
  ]);
  assert.match(elements.get('groups-result').textContent,/4 条 episode/);
  assert.equal(run("JSON.stringify(parseIds(formatIds([5,2,1,0,5])))"),'[0,1,2,5]');
});

test('failed detection preserves existing groups',async()=>{
  const {run}=setup(); const before=run('JSON.stringify(groupInputs())');
  run("api=async()=>{throw Error('missing provenance')}");
  await assert.rejects(run('detectGroups()'),/missing provenance/);
  assert.equal(run('JSON.stringify(groupInputs())'),before);
});

test('loading an existing group preserves its episode order',()=>{
  const {run}=setup();
  run("$('groups').replaceChildren(); addGroup({name:'existing',weight:1,episodes:[5,0,2]})");
  assert.equal(run('groupInputs()[0].episodes'),'5,0,2');
});

for(const mutation of ["fields.get('dataset_root').input.value='/new'",'loadingVersion++',"$('sampling-mode').value='epoch_shuffle'","addGroup({name:'manual',weight:2,episodes:[9]})"]){
  test('in-flight result cannot overwrite changed input: '+mutation,async()=>{
    const {context,run}=setup(); context.result=detected;
    run('let finish; api=()=>new Promise(resolve=>{finish=resolve})');
    const pending=run('detectGroups()'); run(mutation);
    const changed=run('JSON.stringify(groupInputs())'); run('finish(result)');
    await assert.rejects(pending,/请重新读取/);
    assert.equal(run('JSON.stringify(groupInputs())'),changed);
  });
}

test('stop from a task row targets that job, including download/preparation stages',async()=>{
  const {context,run}=setup(); context.calls=[];context.confirm=()=>true;
  run(`currentJob='another-job';
    api=async(path,data)=>{calls.push([path,data]);return path.startsWith('/api/job?')?{status:'running',stage:'下载权重',exp_name:'selected'}:{message:'已请求停止'};};
    refreshJobs=async()=>{};showJob=async id=>{calls.push(['shown',id]);};`);
  await run("stopTask('selected-job')");
  assert.equal(context.calls[1][0],'/api/stop');
  assert.equal(context.calls[1][1].id,'selected-job');
  assert.equal(context.calls[2][1],'selected-job');
});

test('stop cancellation and completed jobs do not send stop requests',async()=>{
  const {context,run}=setup();context.calls=[];context.confirm=()=>false;
  run("api=async(path,data)=>{calls.push(path);return {status:'running',exp_name:'test'};}");
  await run("stopTask('id')");
  assert.equal(context.calls.length,1);
  run("api=async()=>({status:'completed'})");
  await assert.rejects(run("stopTask('id')"),/已结束/);
});

test('parent discovery keeps full selected and subtasks unchecked, then applies only checked roots',async()=>{
  const {context,run,elements}=setup();
  context.discovery={datasets:[
    {root:'/0924/A',episodes:106,frames:10,fps:30,selected:true},
    {root:'/1010/full',episodes:191,frames:20,fps:30,selected:true},
    {root:'/1010/subtasks',episodes:760,frames:20,fps:30,selected:false,note:'alternative'},
  ],warnings:[]};
  run("api=async()=>discovery;fields.get('dataset_root').input.dispatchEvent=()=>{};fields.set('sources',{input:{value:'',dispatchEvent(){}}});independentStats=()=>{};");
  await run('discoverDatasets()');
  assert.equal(elements.get('dataset-candidates').querySelectorAll('input:checked').length,2);
  run('applyDatasets()');
  assert.equal(run("fields.get('dataset_root').input.value"),'/0924/A\n/1010/full');
  assert.equal(run("fields.get('sources').input.value"),'/0924/A\n/1010/full');
  assert.equal(elements.get('groups').children.length,0);
});

test('changed directory cannot apply stale discovery, and browser can append multiple paths',async()=>{
  const {context,run}=setup();context.discovery={datasets:[],warnings:[]};
  run("api=async()=>discovery;initial={dataset_root:'/default'};");
  assert.equal(run("appendDatasetPath('/default','/0924')"),'/0924');
  assert.equal(run("appendDatasetPath('/0924','/1010')"),'/0924\n/1010');
  await run('discoverDatasets()');
  run("fields.get('dataset_root').input.value='/changed'");
  assert.throws(()=>run('applyDatasets()'),/目录已修改/);
});

test('launch request carries every selected root in order while one root keeps legacy shape',()=>{
  const {run}=setup();
  run("$('sampling-mode').value='epoch_shuffle';fields.get('dataset_root').input.value='/a\\n/b';fields.set('owner',{input:{value:'tester'},type:'text'});");
  assert.equal(run('JSON.stringify(request().values.dataset_roots)'),'["/a","/b"]');
  assert.equal(run('request().values.dataset_root'),'/a');
  run("fields.get('dataset_root').input.value='/a'");
  assert.equal(run('request().values.dataset_roots'),undefined);
});

test('summary shows Paper details only for Paper',()=>{
  const {run,elements}=setup();
  elements.set('protocol',new Element('div'));
  run(`initial={action_semantics:'absolute',assets_dir:'/assets',asset_id:'ordinary'};
    protocol={action_dim:58,model_action_dim:58,action_hz:30,model_family:'pi05',prompt:'Pick tomato.'};
    renderProtocol();`);
  let text=elements.get('protocol').children.map(el=>el.textContent).join('\n');
  assert.match(text,/普通 π0.5/);
  assert.doesNotMatch(text,/beta=/);
  assert.match(text,/\/assets\/ordinary/);
  run(`protocol.model_family='Paper';protocol.paper_algorithm={mode:'timestamp_zscore_arms',beta:.5,asset_sha256:'fixture-hash'};renderProtocol();`);
  text=elements.get('protocol').children.map(el=>el.textContent).join('\n');
  assert.match(text,/Paper 专用配置/);
  assert.match(text,/timestamp_zscore_arms.*beta=0.5.*fixture-hash/);
});

test('Paper action selector disables absolute without affecting ordinary baseline',()=>{
  const {run,context}=setup();
  context.container=new Element('div');
  run(`initial={action_semantics:'delta'};protocol={delta_available:true,paper_algorithm:{mode:'timestamp_zscore_arms'}};
    addField(container,['action_semantics','监督','action-semantics'],'delta','delta');`);
  assert.equal(run("fields.get('action_semantics').input.querySelector('[value=absolute]').disabled"),true);
  run(`protocol={delta_available:true};addField(container,['action_semantics','监督','action-semantics'],'absolute','absolute');`);
  assert.notEqual(run("fields.get('action_semantics').input.querySelector('[value=absolute]').disabled"),true);
});

test('GUI checkboxes select T1/T2/T3, lock sampling, rotate assets and restore native mode',()=>{
  const {run,elements,context}=setup();
  for(const id of ['paper-timestamp','paper-correlated','paper-help','protocol','groups-panel'])elements.set(id,new Element('div'));
  elements.get('sampling-mode').append(Object.assign(new Element('option'),{value:'weighted_replacement'}));
  run(`initial={paper_trial:'none',asset_id:'ordinary',assets_dir:'/assets'};
    protocol={paper_available:true,action_dim:58,model_action_dim:58,action_hz:30,prompt:'Pick tomato.'};
    for(const [key,value] of Object.entries({exp_name:'run01',asset_id:'ordinary',assets_dir:'/assets'}))fields.set(key,{input:{value}});
    `);
  const input=new Element('select');input.value='absolute';input.append(Object.assign(new Element('option'),{value:'absolute'}));
  context.inputForTest=input;
  run("fields.set('action_semantics',{input:inputForTest});independentStats=()=>{fields.get('asset_id').input.value='independent_paper_'+paperChoice};");
  elements.get('paper-timestamp').checked=true;
  run('changePaperFeatures()');
  assert.equal(run('paperChoice'),'t1');
  assert.equal(input.value,'delta');
  assert.equal(input.querySelector('[value=absolute]').disabled,true);
  assert.equal(elements.get('sampling-mode').value,'epoch_shuffle');
  assert.equal(run('groupInputs().length'),0);
  assert.equal(run('paperAlgorithm().beta'),0);
  assert.match(run("fields.get('asset_id').input.value"),/paper_t1$/);
  elements.get('paper-correlated').checked=true;run('changePaperFeatures()');
  assert.equal(run('paperChoice'),'t3');assert.equal(run('paperAlgorithm().beta'),.5);
  elements.get('paper-timestamp').checked=false;run('changePaperFeatures()');
  assert.equal(run('paperChoice'),'t2');assert.equal(run('paperAlgorithm().mode'),'global_quantile');
  elements.get('paper-correlated').checked=false;run('changePaperFeatures()');
  assert.equal(run('paperChoice'),'none');assert.equal(run('paperAlgorithm()'),null);
  assert.equal(input.querySelector('[value=absolute]').disabled,false);
  assert.equal(run("fields.get('asset_id').input.value"),'ordinary');
  assert.equal(run('request().values.paper_trial'),'none');
});
