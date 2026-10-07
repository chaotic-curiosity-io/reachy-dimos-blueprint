"""Local browser view of experimental DimOS mapping plus separate Reachy RGB.

Reuses the existing IntelRealSense WebGL renderer; no cloud services or CDN.
"""
import argparse
import json
import mimetypes
import os
from pathlib import Path
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import parse_qs, urlsplit
from station.l515.reachy_perception import recall
from station.l515.reachy_perception import atomic_json

import numpy as np
import open3d as o3d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--port', type=int, default=8777)
    p.add_argument('--reachy-url', default=os.environ.get('REACHY_URL', 'http://reachy-mini.local:8042'),
                   help='wheels app base URL on the robot (env REACHY_URL)')
    args=p.parse_args()
    cache_lock=threading.Lock()
    cache={'mtime':None, 'payload':None}
    panel='''<section id="rgb-panel"><h2>Reachy head camera</h2><img id="rgb-frame" alt="Live Reachy RGB view"><p>Separate camera · not aligned to depth</p><p id="map-quality">Reading localization…</p><p><a href="/map.ply">Download provisional map</a></p><small>Experimental scan matching. Sensor-relative pose, uncalibrated mounting, no loop closure. No autonomous driving.</small></section>
    <style>.panel{left:24px;right:auto;top:105px}#rgb-panel{position:fixed;right:24px;top:105px;width:min(320px,35vw);padding:16px;background:#141c29ed;border:1px solid #344257;border-radius:16px;color:#dae6f4;font:13px system-ui;z-index:3}#rgb-panel h2{font-size:15px;margin:0 0 12px}#rgb-panel img{width:100%;border-radius:8px}#rgb-panel p{line-height:1.5}#rgb-panel a{color:#79d6c0}#rgb-panel small{color:#a6b6c9}@media(max-width:700px){.panel{left:10px;top:85px;width:180px}#rgb-panel{top:auto;bottom:60px;right:10px;width:200px;max-height:42vh;overflow:auto}}</style>
    <script>
    const image=document.getElementById('rgb-frame');
    function nextImage(){image.src='/api/rgb?t='+Date.now()}
    image.onload=()=>setTimeout(nextImage,1500);
    image.onerror=()=>setTimeout(nextImage,3000);
    nextImage();
    async function quality(){try{const q=await(await fetch('/api/localization')).json();document.getElementById('map-quality').textContent=`${q.state} · ${q.accepted} accepted / ${q.rejected} rejected · ${Math.round((q.fitness||0)*100)}% overlap · ${((q.rmse_m||0)*1000).toFixed(1)} mm residual · map updated every 5 s`;}catch(e){document.getElementById('map-quality').textContent='Localization unavailable';}setTimeout(quality,1000)}quality();
    </script>'''
    def localization():
        data=json.loads((args.directory/'continuous-mapping.json').read_text())
        age=time.time()-data['reported_at_unix']
        if not data.get('running') or not 0<=age<2:
            data.update(state='offline', reason='mapper report is stale or stopped')
        return data
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, mime, code=200):
            self.send_response(code)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def do_GET(self):
            path=self.path.split('?',1)[0]
            try:
                if path=='/api/viewer-health':
                    stack=json.loads((args.directory/'stack-status.json').read_text())
                    service=stack.get('rerun',{})
                    ready=False
                    if service.get('running'):
                        try:
                            with urlopen('http://127.0.0.1:8778/',timeout=1) as r:
                                ready=r.status==200
                        except OSError:
                            pass
                    return self.send(json.dumps({'pid':service.get('pid'),'ready':ready}).encode(),'application/json')
                if path=='/api/perception':
                    report=json.loads((args.directory/'perception-live.json').read_text())
                    if time.time()-report['reported_at_unix']>3: report['state']='stale'
                    return self.send(json.dumps(report).encode(),'application/json')
                if path=='/api/map-segments':
                    segments=[]
                    for saved in sorted((args.directory/'colored-map-segments').glob('*.npz')):
                        with np.load(saved,allow_pickle=False) as data:
                            segments.append(json.loads(str(data['report'])))
                    return self.send(json.dumps({'segments':segments,
                        'registration':'separate coordinate frames; not globally aligned'}).encode(),'application/json')
                if path=='/api/observations':
                    query=parse_qs(urlsplit(self.path).query)
                    observations=recall(args.directory/'observations.sqlite3',query.get('name',[None])[0],int(query.get('limit',['50'])[0]))
                    return self.send(json.dumps({'coordinate_space':'per_observation; 2D pixels and optional camera-optical 3D, not map coordinates','observations':observations}).encode(),'application/json')
                if path=='/api/localization':
                    return self.send(json.dumps(localization()).encode(),'application/json')
                if path=='/api/navigation':
                    report=json.loads((args.directory/'navigation-status.json').read_text())
                    if time.time()-report.get('reported_at_unix',0)>2:
                        report.update(running=False,state='offline',blocked_reason='navigation report is stale')
                    return self.send(json.dumps(report).encode(),'application/json')
                if path=='/api/navigation/costmap.png':
                    return self.send((args.directory/'navigation-costmap.png').read_bytes(),'image/png')
                if path=='/api/status':
                    q=localization()
                    state='streaming' if q['state']=='tracking' else q['state']
                    s={'camera':'DimOS L515 map', 'serial':'experimental', 'state':state,
                       'published_fps':0.2,'last_frame_age_ms':(q.get('source_age_s') or 0)*1000,
                       'error':q.get('reason')}
                    return self.send(json.dumps(s).encode(),'application/json')
                if path=='/api/rgb':
                    with urlopen(args.reachy_url+'/api/camera',timeout=3) as r:
                        body=r.read(4*1024*1024+1)
                    if len(body)>4*1024*1024: raise ValueError('oversize image')
                    return self.send(body,'image/jpeg')
                if path=='/map.ply':
                    return self.send((args.directory/'live-map.ply').read_bytes(),'application/octet-stream')
                if path=='/api/points':
                    source=args.directory/'live-map.ply'
                    modified=source.stat().st_mtime_ns
                    with cache_lock:
                        if modified!=cache['mtime']:
                            cloud=o3d.io.read_point_cloud(str(source))
                            points=np.asarray(cloud.points,dtype='<f4')
                            cache['payload']=struct.pack('<4sIII',b'RSPC',1,modified&0xffffffff,len(points))+points.tobytes()
                            cache['mtime']=modified
                        body=cache['payload']
                    return self.send(body,'application/octet-stream')
                if path in ('/','/index.html'):
                    page='<!doctype html><html><head><title>Reachy DimOS · Rerun</title><style>html,body{margin:0;height:100%;background:#111821;color:#e0e8f1;font:14px system-ui}header{padding:10px 16px;height:24px}a{color:#78d6bf;margin-right:24px}iframe{width:100%;height:calc(100% - 44px);border:0}</style></head><body><header>Reachy DimOS · <a href="http://127.0.0.1:8778/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9878%2Fproxy" target="_blank">Open Rerun separately</a><a href="/navigation">Navigation</a><a href="/legacy">Legacy point view</a><a href="/map.ply">Download map</a></header><iframe src="http://127.0.0.1:8778/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9878%2Fproxy" title="Live Rerun RGB points voxels and segmentation"></iframe><script>let currentPid=null;async function health(){try{const r=await fetch("/api/viewer-health",{signal:AbortSignal.timeout(2500)});if(r.ok){const s=await r.json();if(s.ready){if(currentPid!==s.pid){const f=document.querySelector("iframe");f.src=f.src;}currentPid=s.pid;}}}catch(e){}setTimeout(health,2000)}health();</script></body></html>'
                    return self.send(page.encode(),'text/html; charset=utf-8')
                if path=='/navigation':
                    page='''<!doctype html><html><head><title>Reachy DimOS navigation</title><style>body{max-width:900px;margin:30px auto;padding:0 20px;background:#111821;color:#e0e8f1;font:15px system-ui}a{color:#78d6bf}section{background:#182331;border:1px solid #344257;border-radius:14px;padding:18px;margin:16px 0}input,button{font:inherit;padding:8px;margin:4px;background:#243447;color:white;border:1px solid #52677e;border-radius:7px}button.stop{background:#782d38}img{max-width:100%;image-rendering:pixelated;border-radius:8px}pre{white-space:pre-wrap}</style></head><body><p><a href="/">← Rerun</a></p><h1>Reachy · DimOS navigation</h1><section><form id="goal"><label>x (m) <input name="x" type="number" step="0.05" required></label><label>y (m) <input name="y" type="number" step="0.05" required></label><label>yaw° <input name="yaw_deg" type="number" value="0" step="5"></label><button>Go to observed free point</button></form><button id="explore">Explore mapped frontiers</button><button id="stop" class="stop">STOP</button><p>Coordinates use the fixed navigation frame: x forward, y left, z up from the first L515 pose. Unknown space is blocked.</p></section><section><h2>Costmap</h2><img id="map" src="/api/navigation/costmap.png"><p>green robot · blue path · red occupied · light free · dark unknown</p></section><section><h2>Status</h2><pre id="status">loading…</pre></section><script>async function command(body){const r=await fetch('/api/navigation/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const out=await r.json();if(!r.ok)alert(out.error||'request failed')}document.getElementById('goal').onsubmit=e=>{e.preventDefault();const f=new FormData(e.target);command({action:'goal',x:+f.get('x'),y:+f.get('y'),yaw_deg:+f.get('yaw_deg')})};document.getElementById('explore').onclick=()=>command({action:'explore'});document.getElementById('stop').onclick=()=>command({action:'stop'});async function refresh(){try{const s=await(await fetch('/api/navigation')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);document.getElementById('map').src='/api/navigation/costmap.png?t='+Date.now()}catch(e){document.getElementById('status').textContent=String(e)}setTimeout(refresh,1000)}refresh()</script></body></html>'''
                    return self.send(page.encode(),'text/html; charset=utf-8')
                if path=='/legacy':
                    text=(args.assets/'index.html').read_text().replace('Reachy Depth','Reachy · L515 Mapping')
                    text=text.replace('<span>STREAM</span>','<span>MAP REFRESH</span>').replace('<span>LATENCY</span>','<span>POSE AGE</span>')
                    text=text.replace('</body>',panel+'</body>')
                    return self.send(text.encode(),'text/html; charset=utf-8')
                if path in ('/app.js','/styles.css'):
                    return self.send((args.assets/path[1:]).read_bytes(),mimetypes.guess_type(path)[0] or 'text/plain')
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (OSError, ValueError, KeyError) as exc:
                self.send(json.dumps({'error':str(exc)}).encode(),'application/json',503)
        def do_POST(self):
            path=self.path.split('?',1)[0]
            if path!='/api/navigation/command':
                return self.send(json.dumps({'error':'not found'}).encode(),'application/json',404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096: raise ValueError('invalid request size')
                command=json.loads(self.rfile.read(length))
                if command.get('action') not in ('goal','explore','stop'):
                    raise ValueError('action must be goal, explore, or stop')
                if command['action']=='goal':
                    for key in ('x','y'):
                        value=float(command[key])
                        if not np.isfinite(value): raise ValueError(key+' must be finite')
                    command['x'],command['y']=float(command['x']),float(command['y'])
                    command['yaw_deg']=float(command.get('yaw_deg',0))
                    if not np.isfinite(command['yaw_deg']): raise ValueError('yaw_deg must be finite')
                command['requested_at_unix']=time.time()
                atomic_json(args.directory/'navigation-command.json',command)
                return self.send(json.dumps({'ok':True,'command':command}).encode(),'application/json')
            except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                return self.send(json.dumps({'error':str(exc)}).encode(),'application/json',400)
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Experimental mapping viewer: http://127.0.0.1:{args.port}',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__=='__main__':
    main()
