/** Mount an isolated, dependency-free particle-brain renderer. See README.md. */
export function createParticleBrain(container, options = {}) {
  if (!(container instanceof Element)) throw new TypeError('Expected a DOM element');
  const host = document.createElement('div');
  container.append(host);
  const shadow = host.attachShadow({mode: 'open'});
  const styleLink = document.createElement('link');
  styleLink.rel = 'stylesheet';
  styleLink.href = new URL('./particle-brain.css', import.meta.url).href;
  shadow.append(styleLink);
  const template = document.createElement('template');
  template.innerHTML = "<div class=\"particle-brain\">\n<div class=\"caption\"><span>Neural activity</span><span class=\"text-small text-muted\">Simulated \u00b7 drag in any direction</span></div>\n<div class=\"stage\"><canvas role=\"img\" aria-label=\"A floating particle brain with pronounced folds and a softly joined center. In SPAR mode the cerebellum represents Core Beliefs; the left and right cerebral hemispheres represent A1 and A2.\"></canvas></div>\n<div class=\"caption text-small text-muted\"><span>Frontal \u2192 posterior</span><span data-mapping>Cerebellum \u00b7 inactive</span></div>\n<div class=\"viz-controls controls\">\n<label class=\"form-label\">Project<select class=\"form-select\" data-mode><option value=\"hasbro\">Hasbro Prep</option><option value=\"spar\">SPAR</option></select></label>\n<button type=\"button\" class=\"btn btn-primary\" data-generate>Generate answer</button>\n<button type=\"button\" class=\"btn\" data-pause aria-pressed=\"false\">Pause motion</button>\n<button type=\"button\" class=\"btn btn-ghost\" data-view>Rear view</button>\n</div>\n<div class=\"fields\">\n<label class=\"form-label field\"><span class=\"readout\"><span data-core-label>Regional deviation</span><output data-value=\"0\">0%</output></span><input class=\"form-range\" type=\"range\" min=\"0\" max=\"100\" value=\"0\" data-region=\"0\" aria-label=\"Core brain deviation\"></label>\n<label class=\"form-label field\" data-support hidden><span class=\"readout\"><span>A1 \u00b7 left brain</span><output data-value=\"1\">0%</output></span><input class=\"form-range\" type=\"range\" min=\"0\" max=\"100\" value=\"0\" data-region=\"1\" aria-label=\"Supporting belief a1 deviation\"></label>\n<label class=\"form-label field\" data-support hidden><span class=\"readout\"><span>A2 \u00b7 right brain</span><output data-value=\"2\">0%</output></span><input class=\"form-range\" type=\"range\" min=\"0\" max=\"100\" value=\"0\" data-region=\"2\" aria-label=\"Supporting belief a2 deviation\"></label>\n</div>\n<div class=\"text-small\" aria-live=\"polite\" data-status style=\"margin-top:12px\"></div>\n</div>";
  shadow.append(template.content.cloneNode(true));
  const root = shadow.querySelector('.particle-brain');
  if (options.controls === false) {
    root.querySelectorAll('.caption,.controls,.fields,[data-status]').forEach(el => el.hidden = true);
  }
  let destroyed = false, frame = 0, controlled = false, externalProgress = null;

const canvas=root.querySelector('canvas'),ctx=canvas.getContext('2d'),modeControl=root.querySelector('[data-mode]'),generate=root.querySelector('[data-generate]'),pause=root.querySelector('[data-pause]'),view=root.querySelector('[data-view]'),status=root.querySelector('[data-status]');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let seed=712;function random(){seed=(seed*1664525+1013904223)>>>0;return seed/4294967296}const points=[];
// Sculpted particle cortex: coherent winding ridges and deeper, sparse sulci.
for(const side of [-1,1])for(let i=0;i<5400;i++){
const u=random()*Math.PI*2,c=random(),s=Math.sqrt(1-c*c),a=Math.cos(u)*s,b=Math.sin(u)*s;
const phase=11*u+2.3*Math.sin(3*u+5*c)+2.8*Math.sin(7*c+u)+side*.4*c;
const ridge=(.5+.5*Math.cos(phase))**.65;
const sulcus=Math.exp(-Math.pow((a+.05+.24*b)/.045,2))*.10;
const lateral=Math.exp(-Math.pow((b+.29+.14*a)/.043,2))*.09*(.4+.6*c);
const fold=.90+.115*ridge-sulcus-lateral;
const shell=random()<.94?1:.72+random()*.23;
let x=1.49*a*fold*shell,y=1.02*b*fold*shell+.22-.075*a,z=side*.86*c*fold*shell;
// Lower posterior cortex settles over the tucked-in cerebellum.
if(y<-.12)y-=.035*Math.exp(-Math.pow((x-.9)/.5,2));
const brightness=(.32+.68*ridge)*(shell<1?.38:1)*(1-sulcus*5)*(1-lateral*5);
points.push({x,y,z,kind:0,region:side<0?1:2,brightness,seed:random()*6.28,r:.42+random()*.52});}
// A single tucked-in cerebellar envelope; both halves represent Core Beliefs in SPAR.
for(const side of [-1,1])for(let i=0;i<1450;i++){
const u=random()*Math.PI*2,c=random(),s=Math.sqrt(1-c*c),b=Math.sin(u)*s,ridge=(.5+.5*Math.cos(29*b+1.3*Math.sin(3*u)*c))**.6,f=.94+.075*ridge;
points.push({x:.78+.50*Math.cos(u)*s*f,y:-.66+.30*b*f,z:side*.49*c*f,kind:1,region:0,brightness:.35+.65*ridge,seed:random()*6.28,r:.38+random()*.43});}
let w=700,h=440,t=0,last=0,running=false,started=0,paused=reduced.matches,mode='spar',deviations=[0,0,0],smooth=[0,0,0],activity=[0,0,0],angle=-.60,targetAngle=-.60,tilt=.27,targetTilt=.27,rear=false,drag=null;
let blue,red,neutral,foreground;
function color(token){const e=document.createElement('span');e.style.color='var('+token+')';root.append(e);const c=getComputedStyle(e).color;e.remove();return c;}
function colors(){blue=color('--blue');red=color('--red');neutral=color('--muted-foreground');foreground=color('--foreground')}colors();
const themeObserver=new MutationObserver(colors);themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['class','style','data-theme']});
function resize(){const rect=canvas.getBoundingClientRect();w=rect.width;h=rect.height;const d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0)}const sizeObserver=new ResizeObserver(resize);sizeObserver.observe(canvas);
function save(){host.dispatchEvent(new CustomEvent('brainchange',{detail:getState()}))}
function report(){root.querySelectorAll('[data-region]').forEach(e=>{e.value=Math.round(deviations[Number(e.dataset.region)]*100)});root.querySelectorAll('[data-value]').forEach(e=>{e.value=Math.round(deviations[Number(e.dataset.value)]*100)+'%'});root.querySelectorAll('[data-support]').forEach(e=>e.hidden=mode!=='spar');root.querySelector('[data-core-label]').textContent=mode==='spar'?'Core Beliefs · cerebellum':'Regional deviation';root.querySelector('[data-mapping]').textContent=mode==='spar'?'Cerebellum · Core Beliefs':'Cerebellum · inactive';modeControl.value=mode;const bad=deviations.map((d,i)=>d>.35&&(mode==='spar'||i===0)?['core','a1','a2'][i]:null).filter(Boolean);status.textContent=(running?'Generating':'Idle')+' · '+(bad.length?'Deviation: '+(mode==='spar'?bad.join(', '):'cortical region'):'within baseline')+(paused?' · motion paused':'');pause.textContent=paused?'Resume motion':'Pause motion';pause.setAttribute('aria-pressed',String(paused));generate.textContent=running?'Stop generation':'Generate answer';view.textContent=rear?'Three-quarter view':'Rear view'}
const clamp=v=>Math.max(0,Math.min(1,Number(v)||0));
root.querySelectorAll('[data-region]').forEach(e=>e.addEventListener('input',()=>{deviations[Number(e.dataset.region)]=Number(e.value)/100;report();save()}));modeControl.addEventListener('change',()=>{mode=modeControl.value;report();save()});pause.addEventListener('click',()=>{paused=!paused;report();save()});generate.addEventListener('click',()=>{controlled=false;externalProgress=null;running=!running;started=t;report()});view.addEventListener('click',()=>{rear=!rear;targetAngle=rear?-Math.PI/2:-.60;targetTilt=.27;report();save()});
canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,angle:targetAngle,tilt:targetTilt};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;targetAngle=drag.angle+(e.clientX-drag.x)*.007;targetTilt=Math.max(-1.4,Math.min(1.4,drag.tilt+(e.clientY-drag.y)*.007))});canvas.addEventListener('pointerup',()=>{drag=null;save()});canvas.addEventListener('pointercancel',()=>{drag=null});canvas.addEventListener('lostpointercapture',()=>{drag=null});report();
function project(x,y,z){const yaw=angle+Math.sin(t*.23)*.10,roll=-.035+Math.sin(t*.31)*.018,pitch=tilt;
let xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw),yy=y*Math.cos(pitch)-zz*Math.sin(pitch);zz=y*Math.sin(pitch)+zz*Math.cos(pitch);
const nx=xx*Math.cos(roll)-yy*Math.sin(roll),ny=xx*Math.sin(roll)+yy*Math.cos(roll),scale=Math.min(w/3.85,h/3.4),perspective=5.8/(5.8-zz);
return {x:w*.5+nx*scale*perspective,y:h*.45-ny*scale*perspective+Math.sin(t*.8)*4,z:zz,k:perspective}}
function draw(now){if(destroyed)return;const dt=Math.min((now-last)/1000||0,.05);last=now;if(!paused)t+=dt;angle+=(targetAngle-angle)*Math.min(1,dt*7);tilt+=(targetTilt-tilt)*Math.min(1,dt*7);smooth=smooth.map((v,i)=>v+(deviations[i]-v)*Math.min(1,dt*5));if(running&&!controlled&&t-started>9){running=false;report()}
ctx.clearRect(0,0,w,h);const progress=running?(externalProgress??(controlled?((t-started)%9)/9:(t-started)/9)):0,front=-1.7+3.5*progress;
const plotted=points.map(p=>{const q=project(p.x,p.y,p.z),active=p.kind===0||mode==='spar';let wave=running&&active?(p.kind===0?Math.exp(-Math.pow((p.x-front)/.26,2)):Math.exp(-Math.pow((progress-(p.kind===1?.35:.65))/.17,2))):0;const glow=mode==='spar'?activity[p.region]:activity[0];const hot=Math.max(glow,wave*(.5+.5*(.5+.5*Math.sin(t*4+p.seed*11))**6));const local=p.kind===0?(mode==='spar'?1:Math.exp(-Math.pow((p.x-.32)/.39,4))):1;return {p,q,hot,red:active?local*smooth[mode==='spar'?p.region:0]:0,active}}).sort((a,b)=>a.q.z-b.q.z);
for(const a of plotted){const {p,q,hot}=a;const depth=Math.max(.1,Math.min(1,(q.z+1.3)/2.3));const alpha=(.12+.65*depth)*p.brightness+(hot*.37);const r=p.r*q.k*(1+hot*.9);ctx.fillStyle=a.active?blue:neutral;ctx.globalAlpha=Math.min(1,alpha*(a.active?1:.65));ctx.beginPath();ctx.arc(q.x,q.y,r,0,6.283);ctx.fill();
if(a.red>.01){ctx.fillStyle=red;ctx.globalAlpha=Math.min(.95,a.red*(.3+.6*depth)*p.brightness);ctx.beginPath();ctx.arc(q.x,q.y,r+.15,0,6.283);ctx.fill()}
if(hot>.4||(a.red>.5&&p.seed>5.6)){ctx.fillStyle=a.red>.35?red:blue;ctx.globalAlpha=.045*(hot+a.red);ctx.beginPath();ctx.arc(q.x,q.y,r*4,0,6.283);ctx.fill();ctx.globalAlpha=.12*(hot+a.red);ctx.beginPath();ctx.arc(q.x,q.y,r*1.9,0,6.283);ctx.fill()}}
// Label anatomical halves as they rotate.
if(mode==='spar'){
const anchors=[{q:project(0,.3,-.65),text:'A1',id:1},{q:project(0,.3,.65),text:'A2',id:2}].sort((a,b)=>a.q.x-b.q.x);
anchors.forEach((a,i)=>{const x=w*(i===0?.32:.68),y=h-16;ctx.globalAlpha=.4;ctx.strokeStyle=deviations[a.id]>.35?red:neutral;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(a.q.x,a.q.y);ctx.lineTo(x,y-18);ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle=foreground;ctx.font=getComputedStyle(status).font;ctx.textAlign='center';ctx.fillText(a.text,x,y)})}
ctx.globalAlpha=1;frame=requestAnimationFrame(draw)}
setState(options);resize();frame=requestAnimationFrame(draw);

function getState() {
  return {mode, generating:running, progress:externalProgress, deviations:[...deviations], activity:[...activity], paused, yaw:targetAngle, pitch:targetTilt};
}
function setState(next = {}) {
  if (destroyed) return;
  if (next.mode !== undefined) {
    if (!['hasbro','spar'].includes(next.mode)) throw new RangeError('mode must be hasbro or spar');
    mode = next.mode;
  }
  if (next.deviations !== undefined) {
    if (!Array.isArray(next.deviations) || next.deviations.length !== 3) throw new TypeError('deviations must be [core, a1, a2]');
    deviations = next.deviations.map(clamp);
  }
  if (next.activity !== undefined) {
    if (!Array.isArray(next.activity) || next.activity.length !== 3) throw new TypeError('activity must be [core, a1, a2]');
    activity = next.activity.map(clamp);
  }
  if (next.deviation !== undefined) deviations[0] = clamp(next.deviation);
  if (next.progress !== undefined) externalProgress = next.progress === null ? null : clamp(next.progress);
  if (next.paused !== undefined) paused = Boolean(next.paused);
  if (next.generating !== undefined) {
    if (next.generating && !running) started = t;
    running = Boolean(next.generating);
    controlled = true;
  }
  report();
}
function setView({yaw=targetAngle,pitch=targetTilt} = {}) {
  if (destroyed) return;
  if (Number.isFinite(yaw)) targetAngle=yaw;
  if (Number.isFinite(pitch)) targetTilt=Math.max(-1.4,Math.min(1.4,pitch));
}
function onReducedMotion(event) { paused=event.matches; report(); }
reduced.addEventListener('change',onReducedMotion);
styleLink.addEventListener('load',colors);
return {
  element:host, setState, getState, setView, refreshTheme:colors,
  destroy() {
    if (destroyed) return;
    destroyed=true;
    cancelAnimationFrame(frame);
    sizeObserver.disconnect();
    themeObserver.disconnect();
    reduced.removeEventListener('change',onReducedMotion);
    styleLink.removeEventListener('load',colors);
    host.remove();
  }
};
}
