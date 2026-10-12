// Experiment IDs are displayed/counterbalanced A1 and A2, already remapped by design.py.
const index = {C:0,core:0,A1:1,A2:2};
function list(value) {
  if (Array.isArray(value)) return value;
  if (typeof value !== 'string') return [];
  try { const parsed=JSON.parse(value); return Array.isArray(parsed)?parsed:[]; } catch { return []; }
}
const yes=value=>value===true||value==='True'||value==='true';
export function turnToBrainState(row) {
  const activity=[0,0,0];
  const reasons=[];
  for (const id of list(row.active_auxiliaries)) if (index[id]!==undefined) {
    activity[index[id]]=.35; reasons.push(`${id} active`);
  }
  if (yes(row.inject_evidence)) {
    if (String(row.evidence_role||'').startsWith('core_')) {activity[0]=1;reasons.push('Core evidence');}
    for (const id of list(row.evidence_targets)) if (index[id]!==undefined) {
      activity[index[id]]=1;reasons.push(`${id} evidence target`);
    }
  }
  // Probe fields can be carried forward. Only highlight a newly executed probe.
  if (yes(row.probe_ran_after_turn)&&index[row.changed_node]!==undefined) {
    activity[index[row.changed_node]]=1;reasons.push(`Probe reports change: ${row.changed_node}`);
  }
  return {patch:{mode:'spar',generating:false,progress:null,activity,deviations:[0,0,0]},reasons};
}
export function parseConversations(text) {
  const records=text.split(/\r?\n/).filter(line=>line.trim()).map((line,i)=>{
    let record;try {record=JSON.parse(line)}catch{throw new Error(`Invalid JSON on line ${i+1}`)}
    if (!record||!Array.isArray(record.rows)||!record.rows.length||record.rows.some(r=>!r||typeof r!=='object'||Array.isArray(r))) throw new Error(`Line ${i+1} has no valid turn rows. Choose the experiment's dialogue JSONL output.`);
    return record;
  });
  if (!records.length) throw new Error('This file contains no completed conversations.');
  return records;
}
