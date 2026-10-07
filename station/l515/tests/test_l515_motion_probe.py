import io
import json
import sys
import time

import numpy as np
import pytest

from station.l515 import l515_motion_probe as probe


@pytest.fixture
def setup(monkeypatch,tmp_path):
    state=tmp_path/'mapping.json'
    state.write_text(json.dumps({'running':True,'state':'tracking',
        'reported_at_unix':time.time(),'source_age_s':0.1,'sensor_pose_matrix':np.eye(4).tolist()}))
    out=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['probe','--execute','--command','strafe_right',
        '--mapping-status',str(state),'--output',str(out)])
    def get(url,**kwargs):
        responses={'/api/track/status':{'active':False}, '/api/voice/status':{'voice_enabled':False},
                   '/api/status':{'connected':True,'state':{'moving':False},'host':'board','port':80}}
        return io.BytesIO(json.dumps(next(v for k,v in responses.items() if url.endswith(k))).encode())
    monkeypatch.setattr(probe,'urlopen',get)
    monkeypatch.setattr(probe.time,'sleep',lambda _:None)
    calls=[]
    class Client:
        def __init__(self,config):
            assert config.retries==1
        def command(self,*args,**kwargs):
            calls.append(('command',kwargs))
            raise TimeoutError('uncertain command response')
        def stop(self):
            calls.append(('stop',{}))
            return {'ok':True}
        def state(self):
            return {'moving':False}
    monkeypatch.setattr(probe,'WheelsClient',Client)
    return calls,state,out


def test_uncertain_motion_is_never_retried_and_always_stopped(setup):
    calls,state,out=setup
    with pytest.raises(SystemExit):
        probe.main()
    assert [x[0] for x in calls]==['command','stop']
    assert calls[0][1]=={'speed':0.2,'duration':0.15}
    assert 'command_error' in json.loads(out.read_text())


def test_stale_localization_prevents_motion(setup):
    calls,state,out=setup
    s=json.loads(state.read_text());s['state']='lost';state.write_text(json.dumps(s))
    with pytest.raises(RuntimeError,match='tracking fresh'):
        probe.main()
    assert calls==[]


def test_probe_limits_prevent_motion(setup,monkeypatch):
    calls,_,_=setup
    monkeypatch.setattr(sys,'argv',sys.argv+['--duration','2'])
    with pytest.raises(SystemExit):
        probe.main()
    assert calls==[]


def test_no_implicit_execution(setup,monkeypatch):
    calls,_,_=setup
    monkeypatch.setattr(sys,'argv',[v for v in sys.argv if v!='--execute'])
    with pytest.raises(SystemExit):
        probe.main()
    assert calls==[]
