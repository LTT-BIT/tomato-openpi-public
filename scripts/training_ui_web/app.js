'use strict';
const $ = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.get('token')) { sessionStorage.setItem('training-token', fragment.get('token')); history.replaceState(null, '', location.pathname); }
let sourceHash = null;
let baseline = null, initial = null, protocol = null, currentJob = null, gpuData = [], browseTarget = null, browseState = null, loadingVersion = 0;
let paperChoice = null;
const fields = new Map();
const terminal = new Set(['completed', 'failed', 'stopped', 'interrupted']);
const statusNames = {queued:'等待中', running:'运行中', completed:'训练完成', failed:'失败', stopped:'已停止', interrupted:'已中断', orphaned:'管理进程中断，子进程仍运行'};
const definitions = {
  identity: [['owner','配置负责人','text'],['exp_name','实验名称','text']],
  data: [['dataset_root','数据集目录（每行一个，可先添加父目录再识别）','paths'],['repo_id','数据集 repo_id','text'],['dataset_version','数据版本','text'],['sources','数据来源（每行一项）','lines'],['assets_dir','归一化统计父目录','path'],['asset_id','统计名称 asset_id','text']],
  train: [['action_semantics','动作监督','action-semantics'],['batch_size','全局 batch size','int'],['num_train_steps','训练步数','int'],['peak_lr','峰值学习率','float'],['decay_lr','末尾学习率','float'],['warmup_steps','Warmup 步数','int'],['decay_steps','学习率衰减步数','int'],['num_workers','数据 workers','int'],['fsdp_devices','FSDP 分片数','int'],['action_horizon','动作 horizon','int']],
  advanced: [['log_interval','日志间隔','int'],['save_interval','权重保存间隔','int'],['keep_period','长期保留间隔（空=关闭）','optional-int'],['seed','总 seed','int'],['model_seed','模型 seed','int'],['data_seed','采样 seed','int'],['augmentation_seed','增强 seed','int'],['weight_decay','Weight decay','float'],['clip_gradient_norm','梯度裁剪','float'],['save_before_step','分阶段保存切换步数','int'],['save_steps_before','切换前保存步数（逗号分隔）','ints'],['save_interval_after','切换后保存间隔','int']],
  runtime: [['weight_path','基础权重 params 路径 / URL','path'],['project_name','W&B 项目','text'],['stats_batch_size','统计计算 batch size','int']],
};
function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined) el.textContent=text; if(cls) el.className=cls; return el; }
function notice(text, error=false) { $('notice').textContent=text; $('notice').className='notice'+(error?' error':''); }
async function api(path, data) {
  const options={headers:{'X-Training-Token':sessionStorage.getItem('training-token')||''}};
  if(data!==undefined){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(data);}
  const response=await fetch(path,options);const result=await response.json();if(!response.ok)throw Error(result.error||'请求失败');return result;
}
async function perform(callback, button) { if(button)button.disabled=true; try{await callback();}catch(error){notice(error.message,true);}finally{if(button)button.disabled=false;} }
function selectedGPUs(){return [...document.querySelectorAll('#gpus input:checked')].map(input=>input.value);}
function updatePath(){
  if(!baseline||!fields.has('owner')||!initial)return;
  try{
    const values=request().values, runFields=new Set(['exp_name','dataset_root','assets_dir','asset_id','project_name','dataset_roots']);
    const changed=Object.keys(initial).filter(k=>!runFields.has(k)&&JSON.stringify(values[k])!==JSON.stringify(initial[k]));
    const task=baseline.split('.')[2];
    $('save-location').textContent=changed.length
      ? '参数已修改，将新建：training_configs/experiments/'+task+'/template/'+fields.get('owner').input.value+'/'+fields.get('exp_name').input.value+'.py'
      : '复用配置：'+baseline.replaceAll('.','/')+'.py（本次运行单独记录）';
  }catch{$('save-location').textContent='填写完整参数后可预览保存方式。';}
}
function independentStats(){const suffix=fields.get('action_semantics').input.value==='delta'?'_delta':'';fields.get('asset_id').input.value='ui_'+fields.get('owner').input.value+'_'+fields.get('exp_name').input.value+suffix;fields.get('asset_id').input.dispatchEvent(new Event('input'));}
function addField(container, definition, value, original) {
  const [key,label,type]=definition; const wrapper=node('label',label,(['path','paths','lines'].includes(type))?'full':'');
  const input=document.createElement(['lines','paths'].includes(type)?'textarea':type==='action-semantics'?'select':'input');input.id='field-'+key;input.name=key;
  if(type==='action-semantics'){
    for(const [mode,text] of [['absolute','absolute · 原始绝对目标'],['delta','delta · 关节目标减当前状态']]){const option=node('option',text);option.value=mode;input.append(option);}
    if(protocol.paper_algorithm){input.querySelector('[value=absolute]').disabled=true;wrapper.append(node('small','论文配置固定为双臂 delta、双手 absolute；算法由论文功能选择，来源随配置保存。','default'));}
    if(!protocol.delta_available){input.querySelector('[value=delta]').disabled=true;wrapper.append(node('small','当前仅番茄 Wuji / Sharpa 支持 delta。','default'));}
  }
  if(['int','float','optional-int'].includes(type)){input.type='number';input.step=type==='float'?'any':'1';}
  input.value=Array.isArray(value)?value.join(type==='lines'?'\n':', '):(value??'');
  const wrap=['path','paths'].includes(type)?node('div',undefined,'path-input'):wrapper;wrap.append(input);
  if(['path','paths'].includes(type)){const browse=node('button','浏览');browse.type='button';browse.addEventListener('click',()=>perform(()=>openBrowser(input)));wrap.append(browse);wrapper.append(wrap);}
  if(original!==undefined){wrapper.append(node('small','来源值：'+(Array.isArray(original)?original.join(', '):(original??'关闭')),'default'));}
  input.addEventListener('input',()=>{input.classList.toggle('changed',original!==undefined&&input.value!==(Array.isArray(original)?original.join(type==='lines'?'\n':', '):String(original??'')));updatePath();});
  if(key==='dataset_root')input.addEventListener('input',()=>{$('groups-result').textContent='数据目录已变更，请重新读取或核对分组编号。';});
  if(key==='action_horizon'){input.addEventListener('change',()=>{if(Number(input.value)!==initial.action_horizon)independentStats();});}
  if(key==='action_semantics')input.addEventListener('change',()=>{
    const old=input.dataset.previous||initial.action_semantics, name=fields.get('exp_name').input;
    if(name.value.includes('_'+old+'_'))name.value=name.value.replace('_'+old+'_','_'+input.value+'_');
    else if(name.value.endsWith('_'+old))name.value=name.value.slice(0,-old.length)+input.value;
    name.dispatchEvent(new Event('input'));input.dataset.previous=input.value;input.dispatchEvent(new Event('input'));independentStats();
  });
  if(['action_semantics','assets_dir','asset_id'].includes(key))input.addEventListener('input',renderProtocol);
  fields.set(key,{input,type});container.append(wrapper);
}
function fillForm(result) {
  $('dataset-candidates').replaceChildren();$('apply-datasets').classList.add('hidden');
  $('groups-result').textContent='';
  initial=result.values;sourceHash=result.source_sha256;protocol=result.protocol;fields.clear();
  for(const [section,defs] of Object.entries(definitions)){
    const container=$(section+'-fields');container.replaceChildren();
    for(const definition of defs){const key=definition[0];if(section==='advanced'&&!(key in initial))continue;let value=initial[key];if(key==='dataset_root')value=(initial.dataset_roots||[initial.dataset_root]).join('\n');if(key==='owner')value=result.owner;if(key==='exp_name')value=result.suggested_exp_name;if(key==='stats_batch_size')value=512;addField(container,definition,value,initial[key]);}
  }
  paperChoice=initial.paper_trial||'none';
  if(result.shared_weights){const button=node('button','使用服务器公共权重','text-button');button.addEventListener('click',()=>{fields.get('weight_path').input.value=result.shared_weights;fields.get('weight_path').input.dispatchEvent(new Event('input'));});$('runtime-fields').append(button);}
  $('sampling-mode').value=initial.sampling_mode;$('groups').replaceChildren();for(const group of initial.sampling_groups)addGroup(group);
  $('groups-panel').classList.toggle('hidden',initial.sampling_mode!=='weighted_replacement');
  syncPaperControls();renderProtocol();
  fields.get('exp_name').input.dispatchEvent(new Event('input'));updatePath();
}
function paperAlgorithm(){
  if(paperChoice===null)return protocol.paper_algorithm;
  if(paperChoice==='none')return null;
  const modes={t0:['global_quantile',0],t1:['timestamp_zscore_arms',0],t2:['global_quantile',.5],t3:['timestamp_zscore_arms',.5],tg:['global_zscore_arms',0]};
  const pair=modes[paperChoice];if(!pair)return protocol.paper_algorithm;
  const same=paperChoice===initial.paper_trial&&fields.get('asset_id')?.input.value===initial.asset_id&&fields.get('assets_dir')?.input.value===initial.assets_dir;
  return {mode:pair[0],beta:pair[1],asset_sha256:same?protocol.paper_algorithm?.asset_sha256:null};
}
function syncPaperControls(){
  const active=paperChoice!=='none', available=protocol.paper_available&&paperChoice!=='custom';
  $('paper-timestamp').checked=['t1','t3'].includes(paperChoice);
  $('paper-correlated').checked=['t2','t3'].includes(paperChoice);
  $('paper-timestamp').disabled=!available;$('paper-correlated').disabled=!available;
  $('paper-help').textContent=available?'两项同时勾选为 T3。论文功能使用双臂 delta、双手 absolute、完整动作窗口及无放回采样。切换功能自动分配独立统计。':'论文勾选仅支持 30 Hz Sharpa multimodal π0.5 全参；已有自定义论文配置保留原算法。';
  fields.get('action_semantics').input.querySelector('[value=absolute]').disabled=active;
  $('sampling-mode').querySelector('[value=weighted_replacement]').disabled=active;
  if(active){
    fields.get('action_semantics').input.value='delta';
    $('sampling-mode').value='epoch_shuffle';$('groups').replaceChildren();$('groups-panel').classList.add('hidden');
  }
}
function changePaperFeatures(){
  if(!protocol.paper_available)throw Error('当前配置不支持论文功能');
  paperChoice=$('paper-timestamp').checked?($('paper-correlated').checked?'t3':'t1'):($('paper-correlated').checked?'t2':'none');
  syncPaperControls();
  const name=fields.get('exp_name').input;
  if(paperChoice!=='none')name.value=name.value.replace(/_absolute(?=_|$)/,'_delta');
  name.value=name.value.replace(/_paper_(t[0123g]|none)$/,'')+'_paper_'+paperChoice;
  if(paperChoice===initial.paper_trial){fields.get('asset_id').input.value=initial.asset_id;}
  else{independentStats();}
  renderProtocol();updatePath();
}
function renderProtocol(){
  const current=key=>fields.get(key)?.input.value??initial[key], p=paperAlgorithm();
  $('protocol').replaceChildren(node('div',`${protocol.action_dim} 维动作 · 模型 ${protocol.model_action_dim} 维 · ${protocol.action_hz} Hz · ${p?'Paper 专用配置':'普通 π0.5'} · 动作监督 ${current('action_semantics')}`),node('div','统计资产：'+current('assets_dir')+'/'+current('asset_id')),node('div','英文 prompt 示例：'+protocol.prompt));
  if(p)$('protocol').append(node('div',`Paper mode=${p.mode} · beta=${p.beta} · bundle SHA256=${p.asset_sha256||'待计算'}`));
  if(protocol.task_view)$('protocol').append(node('div','保留来源配置的任务筛选 / prompt 规则（见最终配置）。柠檬数据沿用已记录的 209 条筛选。'));
}
async function chooseBaseline(module){
  const version=++loadingVersion;$('start').disabled=true;
  try{const result=await api('/api/baseline?id='+encodeURIComponent(module));if(version!==loadingVersion)return;baseline=module;fillForm(result);for(const el of document.querySelectorAll('#baseline-list button, #personal-list button'))el.classList.toggle('selected',el.dataset.id===module);}
  finally{if(version===loadingVersion)$('start').disabled=false;}
}
function parseIds(text){
  if(!text.trim())return [];
  const result=[];for(const part of text.split(/[,，\s]+/).filter(Boolean)){
    if(/^\d+$/.test(part)){result.push(Number(part));continue;}
    const range=/^(\d+)-(\d+)$/.exec(part);if(!range)throw Error('episode 编号需为整数、逗号分隔或区间，例如 0,2,4-8');const start=Number(range[1]),end=Number(range[2]);if(end<start||end-start>100000)throw Error('episode 区间无效');for(let i=start;i<=end;i++)result.push(i);
  }return result;
}
function appendDatasetPath(current,path){const original=(initial.dataset_roots||[initial.dataset_root]).join('\n');return current===original?path:[...new Set([...datasetPaths(current),path])].join('\n');}
function datasetPaths(text){return text.split('\n').map(p=>p.trim()).filter(Boolean);}
async function discoverDatasets(){
  if(!baseline)throw Error('请先选择配置');
  const version=loadingVersion, input=fields.get('dataset_root').input, original=input.value;
  const result=await api('/api/discover-datasets',{roots:datasetPaths(original)});
  if(version!==loadingVersion||input.value!==original)throw Error('目录或配置已修改，请重新识别');
  $('dataset-candidates').replaceChildren();
  for(const item of result.datasets){
    const label=node('label',undefined,'check'),check=node('input');check.type='checkbox';check.value=item.root;check.checked=item.selected;
    label.append(check,node('span',`${item.root} · ${item.episodes} 条 / ${item.frames} 帧 · ${item.fps} Hz${item.note?' · '+item.note:''}`));$('dataset-candidates').append(label);
  }
  $('apply-datasets').classList.remove('hidden');$('dataset-candidates').dataset.input=original;
  notice('已发现 '+result.datasets.length+' 个数据集，请核对勾选后点击“应用选中数据集”。'+result.warnings.join('；'));
}
function applyDatasets(){
  if(fields.get('dataset_root').input.value!==$('dataset-candidates').dataset.input)throw Error('目录已修改，请重新识别');
  const roots=[...$('dataset-candidates').querySelectorAll('input:checked')].map(el=>el.value);
  if(!roots.length)throw Error('请至少勾选一个数据集');
  fields.get('dataset_root').input.value=roots.join('\n');fields.get('dataset_root').input.dispatchEvent(new Event('input'));
  fields.get('sources').input.value=roots.join('\n');fields.get('sources').input.dispatchEvent(new Event('input'));
  independentStats();$('groups').replaceChildren();
  if($('sampling-mode').value==='weighted_replacement')notice('已更新数据组合并清空旧分组，请点击“从数据读取分组”设置权重。');
  else notice('已应用 '+roots.length+' 个数据集，将联合计算独立归一化统计。请核对数据版本和 repo_id。');
  $('apply-datasets').classList.add('hidden');$('dataset-candidates').replaceChildren();updatePath();
}
function formatIds(ids){
  const sorted=[...new Set(ids)].sort((a,b)=>a-b), ranges=[];
  for(let i=0;i<sorted.length;i++){const start=sorted[i];let end=start;while(i+1<sorted.length&&sorted[i+1]===end+1)end=sorted[++i];ranges.push(start===end?String(start):start+'-'+end);}
  return ranges.join(',');
}
function groupInputs(){return [...$('groups').children].map(row=>Object.fromEntries([...row.querySelectorAll('input')].map(input=>[input.dataset.key,input.value])));}
async function detectGroups(){
  if(!baseline)throw Error('请先选择配置');
  const module=baseline, version=loadingVersion, root=fields.get('dataset_root').input.value.trim(), previous=groupInputs();
  const mode=$('sampling-mode').value;
  $('groups-result').textContent='正在读取并核对转换来源记录…';
  try{
    const roots=datasetPaths(root);const result=await api('/api/sampling-groups',{baseline:module,dataset_root:roots[0],dataset_roots:roots});
    if(version!==loadingVersion||baseline!==module||fields.get('dataset_root').input.value.trim()!==root||$('sampling-mode').value!==mode||JSON.stringify(groupInputs())!==JSON.stringify(previous))throw Error('读取期间配置、目录或分组已修改，请重新读取。');
    const weights=new Map(previous.map(g=>[g.name.trim(),Number(g.weight)]));
    $('groups').replaceChildren();
    for(const group of result.groups){const weight=weights.get(group.name);addGroup({...group,episode_text:formatIds(group.episodes),weight:Number.isFinite(weight)&&weight>0?weight:1});}
    $('groups-result').textContent=`已核对 ${result.episodes} 条 episode / ${result.frames} 帧。`+result.groups.map(g=>`${g.name}：${g.episodes.length} 条 / ${g.frames} 帧（来源 ${g.source_root}）`).join('；')+'。请确认各组采样权重。';
    updatePath();notice('已从转换来源填入分组；同名组保留原权重，新组权重为 1。');
  }catch(error){if(version===loadingVersion)$('groups-result').textContent='未应用识别结果，现有分组保留。';throw error;}
}
function addGroup(group={name:'',weight:1,episodes:[]}){
  const row=node('div',undefined,'group');for(const [key,label,value] of [['name','组名',group.name],['weight','权重',group.weight],['episodes','episode 编号 / 区间',group.episode_text??group.episodes.join(',')]]){
    const wrapper=node('label',label),input=node('input');input.dataset.key=key;input.value=value;input.addEventListener('input',updatePath);if(key==='weight'){input.type='number';input.step='any';}wrapper.append(input);row.append(wrapper);
  }const remove=node('button','移除');remove.addEventListener('click',()=>{row.remove();updatePath();});row.append(remove);$('groups').append(row);
}
function request(){
  if(!baseline)throw Error('请先选择配置');const values={};let owner,statsBatch;
  for(const [key,{input,type}] of fields){let value=input.value.trim();if(type==='int'||type==='float')value=value===''?NaN:Number(value);if(type==='optional-int')value=value===''?null:Number(value);if(type==='lines')value=value.split('\n').map(v=>v.trim()).filter(Boolean);if(type==='ints')value=parseIds(value);if(typeof value==='number'&&!Number.isFinite(value))throw Error(key+' 请输入有效数值');if(key==='owner')owner=value;else if(key==='stats_batch_size')statsBatch=value;else values[key]=value;}
  const roots=datasetPaths(values.dataset_root);values.dataset_root=roots[0]||'';if(roots.length>1)values.dataset_roots=roots;
  if(paperChoice!==null)values.paper_trial=paperChoice;
  values.sampling_mode=$('sampling-mode').value;values.sampling_groups=[];
  if(values.sampling_mode==='weighted_replacement')for(const row of $('groups').children){values.sampling_groups.push({name:row.querySelector('[data-key=name]').value.trim(),weight:Number(row.querySelector('[data-key=weight]').value),episodes:parseIds(row.querySelector('[data-key=episodes]').value)});}
  return {baseline,source_sha256:sourceHash,owner,values,gpus:selectedGPUs(),wandb_mode:$('wandb-mode').value,auto_norm:$('auto-norm').checked,allow_busy:$('allow-busy').checked,stats_batch_size:statsBatch};
}
async function refreshGPUs(){const old=new Set(selectedGPUs());const result=await api('/api/gpus');gpuData=result.gpus;$('gpus').replaceChildren();if(result.error){$('gpus').textContent=result.error;return;}
  for(const gpu of gpuData){const card=node('div',undefined,'gpu'),label=node('label',undefined,'check'),input=node('input');input.type='checkbox';input.value=gpu.uuid;input.checked=old.has(gpu.uuid);label.append(input,node('span',`GPU ${gpu.index} · ${gpu.name}`));card.append(label);const meter=node('div',undefined,'meter'),progress=node('progress');progress.max=gpu.total_mb||1;progress.value=gpu.used_mb||0;meter.append(progress);card.append(meter,node('small',`显存 ${gpu.used_mb??'?'} / ${gpu.total_mb??'?'} MiB · 利用率 ${gpu.utilization??'?'}%`));$('gpus').append(card);}
}
async function stopTask(id){
  if(!id)throw Error('请先在任务记录中选择运行中的任务');
  const job=await api('/api/job?id='+id);
  if(terminal.has(job.status))throw Error('该任务已结束');
  if(!confirm('停止任务 '+job.exp_name+'？准备/下载过程也会停止，只保留已保存的 checkpoint。'))return;
  const result=await api('/api/stop',{id});notice(result.message);
  await refreshJobs();await showJob(id);
}
async function refreshJobs(){
  const records=await api('/api/jobs');$('jobs').replaceChildren();if(!records.length)$('jobs').textContent='还没有任务。';
  for(const job of records){
    const row=node('div',undefined,'job-row'),button=node('button',job.exp_name,'job');
    button.append(node('small',`${statusNames[job.status]||job.status} · ${job.stage||''} · ${job.created_at}`));
    button.addEventListener('click',()=>perform(async()=>{await showJob(job.id);$('output-section').scrollIntoView({behavior:'smooth',block:'start'});}));row.append(button);
    if(!terminal.has(job.status)){const stop=node('button','停止任务','danger');stop.addEventListener('click',()=>perform(()=>stopTask(job.id),stop));row.append(stop);}
    $('jobs').append(row);
  }
}
async function showJob(id){currentJob=id;const job=await api('/api/job?id='+id);if(currentJob!==id)return;$('stop-active').disabled=terminal.has(job.status);$('active-task').textContent=job.exp_name+' · '+(statusNames[job.status]||job.status);$('output-title').textContent=job.exp_name;$('output').textContent=job.log||'后台任务准备中…';$('job-status').textContent=(statusNames[job.status]||job.status)+' · '+job.stage;$('job-status').classList.remove('hidden');$('stop').classList.toggle('hidden',terminal.has(job.status));$('resume').classList.toggle('hidden',!terminal.has(job.status)||job.status==='completed');$('job-info').textContent=`配置：${job.config_path}\n权重：${job.checkpoint_dir}`+(job.error?'\n错误：'+job.error:'');}
async function showPreview(check){
  currentJob=null;$('stop-active').disabled=true;$('active-task').textContent='在任务记录中选择要停止的任务。';
  $('output-title').textContent=check?'数据与统计检查':'最终训练配置';
  $('output').textContent=check?'正在校验数据目录、元信息和归一化统计…':'正在生成最终配置…';
  $('job-info').textContent='请等待检查结果。';
  for(const id of ['job-status','stop','resume'])$(id).classList.add('hidden');
  $('output-section').scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const result=await api(check?'/api/check':'/api/preview',request());
    $('output').textContent=JSON.stringify(result,null,2);
    $('job-info').textContent=(result.creates_config?'将新建个人配置：':'复用配置：')+result.config_path;
    notice(check?'数据检查完成。'+result.statistics:'配置已展开，可在下方核对。');
  }catch(error){
    $('output').textContent=(check?'数据与统计检查失败：':'配置预览失败：')+error.message;
    $('job-info').textContent='请修正表单后重新检查。';
    throw error;
  }
}
async function openBrowser(input){browseTarget=input;try{await browse(input.value.includes('://')?'':input.value.split('\n').filter(Boolean).at(-1));}catch{await browse('');}$('browser').showModal();}
async function browse(path){browseState=await api('/api/browse?path='+encodeURIComponent(path));$('browse-path').value=browseState.path;$('browse-list').replaceChildren();for(const entry of browseState.entries){const button=node('button','▸ '+entry.name);button.addEventListener('click',()=>perform(()=>browse(entry.path)));$('browse-list').append(button);}if(browseState.truncated)$('browse-list').append(node('p','仅显示前 500 个目录，请直接输入路径'));}
$('preview').addEventListener('click',()=>perform(()=>showPreview(false),$('preview')));
for(const id of ['paper-timestamp','paper-correlated'])$(id).addEventListener('change',()=>perform(changePaperFeatures));
$('check').addEventListener('click',()=>perform(()=>showPreview(true),$('check')));
$('start').addEventListener('click',()=>perform(async()=>{const job=await api('/api/start',request());notice((job.creates_config?'已保存新配置':'已复用原配置')+'，本次实验已记录；数据检查通过后进入训练。');await refreshCatalog();await refreshJobs();await showJob(job.id);$('output-section').scrollIntoView({behavior:'smooth'});},$('start')));
$('reset').addEventListener('click',()=>perform(()=>chooseBaseline(baseline),$('reset')));
$('new-stats').addEventListener('click',independentStats);
$('sampling-mode').addEventListener('change',()=>{const enabled=$('sampling-mode').value==='weighted_replacement';$('groups-panel').classList.toggle('hidden',!enabled);if(enabled&&!$('groups').children.length){addGroup({name:'A',weight:7,episodes:[]});addGroup({name:'B',weight:3,episodes:[]});}updatePath();});
$('discover-datasets').addEventListener('click',()=>perform(discoverDatasets,$('discover-datasets')));
$('apply-datasets').addEventListener('click',()=>perform(async()=>applyDatasets(),$('apply-datasets')));
$('detect-groups').addEventListener('click',()=>perform(detectGroups,$('detect-groups')));
$('add-group').addEventListener('click',()=>addGroup());
$('refresh-gpus').addEventListener('click',()=>perform(refreshGPUs,$('refresh-gpus')));
$('select-free').addEventListener('click',()=>{for(const input of document.querySelectorAll('#gpus input')){const gpu=gpuData.find(g=>g.uuid===input.value);input.checked=gpu.used_mb!==null&&gpu.utilization!==null&&gpu.used_mb<=1024&&gpu.utilization<=5;}});
$('refresh-jobs').addEventListener('click',()=>perform(refreshJobs,$('refresh-jobs')));
for(const id of ['stop','stop-active'])$(id).addEventListener('click',()=>perform(()=>stopTask(currentJob),$(id)));
$('resume').addEventListener('click',()=>perform(async()=>{const selected=selectedGPUs();const payload={id:currentJob,allow_busy:$('allow-busy').checked};if(selected.length)payload.gpus=selected;const job=await api('/api/resume',payload);notice('使用原配置重试 / 续训，当前表单的参数修改不会应用到旧任务。');await refreshJobs();await showJob(job.id);},$('resume')));
$('browse-close').addEventListener('click',()=>$('browser').close());
$('browse-go').addEventListener('click',()=>perform(()=>browse($('browse-path').value)));
$('browse-parent').addEventListener('click',()=>perform(()=>browse(browseState.parent)));
$('browse-select').addEventListener('click',()=>{browseTarget.value=browseTarget.id==='field-dataset_root'?appendDatasetPath(browseTarget.value,browseState.path):browseState.path;browseTarget.dispatchEvent(new Event('input'));browseTarget.dispatchEvent(new Event('change'));$('browser').close();});
async function refreshCatalog(){
  const info=await api('/api/baselines');
  $('runtime').textContent=info.repository+'\nPython: '+info.python;
  for(const [id,entries] of [['baseline-list',info.baselines],['personal-list',info.personal||[]]]){
    $(id).replaceChildren();
    for(const entry of entries){const button=node('button',entry.label,'baseline');button.dataset.id=entry.id;button.classList.toggle('selected',baseline===entry.id);button.addEventListener('click',()=>perform(()=>chooseBaseline(entry.id)));$(id).append(button);}
    if(!entries.length)$(id).textContent='暂无本地个人配置。';
  }
  return info;
}
$('refresh-configs').addEventListener('click',()=>perform(refreshCatalog,$('refresh-configs')));
perform(async()=>{const info=await refreshCatalog();if(info.baselines.length)await chooseBaseline(info.baselines.find(e=>e.id.includes('tomato_wuji')&&e.id.includes('lora'))?.id||info.baselines[0].id);await Promise.all([refreshGPUs(),refreshJobs()]);});
let polling=false;setInterval(async()=>{if(polling||document.hidden)return;polling=true;try{if(currentJob)await showJob(currentJob);await refreshJobs();}catch(error){notice('连接中断：'+error.message,true);}finally{polling=false;}},5000);
