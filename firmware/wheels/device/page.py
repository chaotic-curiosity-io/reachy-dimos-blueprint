"""Browser control page served at GET /.

Buttons and keys are press-and-hold: holding re-sends the command every 300ms
against the server's deadman timer, and releasing sends an explicit stop. If
the tab closes or the wifi drops mid-hold, the deadman stops the wheels.
"""

PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Mecanum Chassis</title><style>
*{box-sizing:border-box}
body{background:#14161a;color:#e6e8eb;font:15px/1.4 system-ui,sans-serif;
margin:0;padding:20px;display:flex;flex-direction:column;align-items:center;gap:16px}
h1{font-size:16px;font-weight:600;margin:0;color:#9aa4b2;letter-spacing:.04em}
#pad{display:grid;grid-template-columns:repeat(3,72px);gap:8px}
button{background:#232830;color:#e6e8eb;border:1px solid #333a45;border-radius:10px;
height:72px;font-size:22px;cursor:pointer;user-select:none;-webkit-user-select:none;
touch-action:manipulation;transition:background .08s}
button:active,button.on{background:#3b6ea5;border-color:#4d87c4}
#stop{background:#7a2532;border-color:#993140;font-size:15px}
#stop:active{background:#a33344}
.row{display:flex;gap:8px;align-items:center}
#spd{width:220px}
#out{font:12px ui-monospace,monospace;color:#8b95a3;white-space:pre;
background:#1b1f25;border:1px solid #2a3039;border-radius:8px;padding:10px;
min-width:320px;min-height:76px}
</style></head><body>
<h1>MECANUM CHASSIS</h1>
<div id=pad>
  <button data-c=diagonal_rl>&#8601;</button>
  <button data-c=forward data-k=w>&#8593;</button>
  <button data-c=diagonal_fr>&#8599;</button>
  <button data-c=strafe_left data-k=a>&#8592;</button>
  <button id=stop>STOP</button>
  <button data-c=strafe_right data-k=d>&#8594;</button>
  <button data-c=diagonal_fl>&#8598;</button>
  <button data-c=reverse data-k=s>&#8595;</button>
  <button data-c=diagonal_rr>&#8600;</button>
</div>
<div class=row>
  <button data-c=rotate_ccw data-k=q style="width:72px">&#8634;</button>
  <button data-c=rotate_cw data-k=e style="width:72px">&#8635;</button>
</div>
<div class=row><span>speed</span><input id=spd type=range min=0.2 max=1 step=0.05 value=0.6>
<span id=spdv>0.60</span></div>
<div id=out>idle</div>
<script>
const spd=document.getElementById('spd'),out=document.getElementById('out'),
      spdv=document.getElementById('spdv');
let timer=null,active=null;
spd.oninput=()=>spdv.textContent=(+spd.value).toFixed(2);

async function post(body){
  try{
    const r=await fetch('/cmd',{method:'POST',body:JSON.stringify(body)});
    const j=await r.json();
    const w=j.state?j.state.wheels:{};
    out.textContent=(j.command||j.error||'?')+'  speed '+(+spd.value).toFixed(2)+
      '\\nFL '+f(w.front_left)+'   FR '+f(w.front_right)+
      '\\nRL '+f(w.rear_left)+'   RR '+f(w.rear_right)+
      '\\nstops in '+(j.state&&j.state.stops_in!=null?j.state.stops_in.toFixed(1)+'s':'-');
  }catch(e){out.textContent='link lost - deadman will stop the wheels'}
}
const f=v=>(v==null?'----':(v>=0?'+':'')+v.toFixed(2));

function begin(cmd,el){
  if(active===cmd)return;
  end();active=cmd;if(el)el.classList.add('on');
  const send=()=>post({command:cmd,speed:+spd.value});
  send();timer=setInterval(send,300);
}
function end(){
  if(timer){clearInterval(timer);timer=null}
  if(active){active=null;post({command:'stop'})}
  document.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
}
document.querySelectorAll('#pad button[data-c],.row button[data-c]').forEach(b=>{
  const c=b.dataset.c;
  b.addEventListener('pointerdown',e=>{e.preventDefault();begin(c,b)});
  ['pointerup','pointerleave','pointercancel'].forEach(ev=>b.addEventListener(ev,end));
});
document.getElementById('stop').onclick=()=>{end();post({command:'stop'})};
const keys={};document.querySelectorAll('[data-k]').forEach(b=>keys[b.dataset.k]=b);
addEventListener('keydown',e=>{const b=keys[e.key.toLowerCase()];
  if(b&&!e.repeat)begin(b.dataset.c,b); if(e.key===' ')end()});
addEventListener('keyup',e=>{if(keys[e.key.toLowerCase()])end()});
addEventListener('blur',end);
</script></body></html>"""
