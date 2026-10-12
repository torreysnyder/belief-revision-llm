import {createParticleBrain} from './particle-brain.js?v=3';
import {parseConversations,turnToBrainState} from './results-adapter.js';
const el=id=>document.getElementById(id);
const brain=createParticleBrain(el('brain'),{mode:'spar',controls:false});
let records=[],selected=0,turn=0,load=0,playTimer=null;
function stopReplay(){
 clearInterval(playTimer);playTimer=null;
 el('autoplay').textContent='Auto-play · 3s';
 el('autoplay').setAttribute('aria-pressed','false');
}
el('autoplay').addEventListener('click',()=>{
 if(playTimer!==null){stopReplay();return;}
 if(!records.length)return;
 el('autoplay').textContent='Pause replay';
 el('autoplay').setAttribute('aria-pressed','true');
 playTimer=setInterval(()=>{turn=(turn+1)%records[selected].rows.length;render();},3000);
});
function render(){
 const record=records[selected],row=record.rows[turn],{patch,reasons}=turnToBrainState(row);
 brain.setState(patch);
 el('summary').textContent=`${row.phase||'Unknown phase'} · brain highlights match the selected turn`;
 el('position').textContent=`Turn ${turn+1} / ${record.rows.length}`;
 el('reasons').textContent=reasons.join(' · ')||'No region activity recorded for this turn.';
 const score=v=>typeof v==='number'&&Number.isFinite(v)?v:'not measured';
 el('beliefs').textContent=`Assistant probe scores (may be carried forward): Core ${score(row.belief_core)} · A1 ${score(row.belief_A1)} · A2 ${score(row.belief_A2)}`;
 el('claims').textContent=`Core: ${record.primary_cell?.vignette?.core_claim||'not recorded'}\nA1: ${row.display_A1_text||'not recorded'}\nA2: ${row.display_A2_text||'not recorded'}`;
 const transcript=el('transcript');
 transcript.replaceChildren();
 record.rows.slice(0,turn+1).forEach((entry,i)=>{
   const exchange=document.createElement('section');
   exchange.className='exchange'+(i===turn?' current':'');
   const heading=document.createElement('div');heading.className='turn-heading';
   heading.textContent=`Turn ${i+1} · ${entry.phase||'Unknown phase'}`;
   exchange.append(heading);
   for(const [role,title,text] of [['user','Simulated user',entry.user_turn],['assistant','Assistant',entry.assistant_turn]]){
     const message=document.createElement('div');message.className='message '+role;
     const speaker=document.createElement('div');speaker.className='speaker';speaker.textContent=title;
     const bubble=document.createElement('div');bubble.className='bubble';bubble.textContent=text||'[No message recorded]';
     message.append(speaker,bubble);exchange.append(message);
   }
   transcript.append(exchange);
 });
 const current=transcript.lastElementChild;
 if(current)transcript.scrollTop=current.offsetTop-transcript.firstElementChild.offsetTop;
 el('autoplay').disabled=false;
 el('previous').disabled=turn===0;el('next').disabled=turn===record.rows.length-1;
}
async function loadResults(readText){
 stopReplay();
 const request=++load;el('error').textContent='';
 try{const parsed=parseConversations(await readText());if(request!==load)return;records=parsed;selected=0;turn=0;
 el('conversation').replaceChildren(...records.map((r,i)=>new Option(`${r.cell_id||r.vignette_id||'Conversation'} · replicate ${r.replicate_index??i}`,String(i))));el('conversation').disabled=false;render();
 }catch(error){if(request===load)el('error').textContent=error.message;}
}
el('file').addEventListener('change',event=>{
 const file=event.target.files[0];if(file)loadResults(()=>file.text());
});
el('latest').addEventListener('click',()=>loadResults(async()=>{
 const response=await fetch('../../dialogues_full_crossed.jsonl',{cache:'no-store'});
 if(!response.ok)throw new Error(`Could not load the latest local run (${response.status}). Use Open results to choose a saved JSONL file.`);
 return response.text();
}));
el('conversation').addEventListener('change',()=>{stopReplay();selected=Number(el('conversation').value);turn=0;render()});
el('previous').addEventListener('click',()=>{stopReplay();if(turn>0){turn--;render()}});
el('next').addEventListener('click',()=>{stopReplay();if(turn<records[selected].rows.length-1){turn++;render()}});
el('pause').addEventListener('click',()=>{brain.setState({paused:!brain.getState().paused});el('pause').textContent=brain.getState().paused?'Resume motion':'Pause motion'});
el('pause').textContent=brain.getState().paused?'Resume motion':'Pause motion';
window.addEventListener('pagehide',()=>{stopReplay();brain.destroy()},{once:true});
