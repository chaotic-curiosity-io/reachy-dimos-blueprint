"""Fixed-intrinsics stereo fit, with leave-one-pair-out cross-camera validation."""
from pathlib import Path
import json
import os
import cv2
import numpy as np
# Data/output folder: next to this script by default; override with CALIBRATION_DIR.
root=Path(os.environ.get('CALIBRATION_DIR') or Path(__file__).parent)
paths=sorted((root/'paired').glob('*/observation.json'))
obs=[json.loads(p.read_text()) for p in paths]
candidate=json.loads((root/'reachy-rgb-intrinsics-candidate.json').read_text())
K2=np.array(candidate['K']);D2=np.array(candidate['D'])
i=obs[0]['metadata']['color_intrinsics']
assert i['model']=='distortion.brown_conrady'
K1=np.array([[i['fx'],0,i['ppx']],[0,i['fy'],i['ppy']],[0,0,1.]])
D1=np.array(i['coeffs'])
b=cv2.aruco.CharucoBoard((5,7),.0235,.01175,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
objects=[];p1=[];p2=[];independent=[]

def pose(obj,pts,K,D):
 ok,r,t=cv2.solvePnP(obj,pts,K,D)
 if not ok:raise ValueError('PnP failed')
 return cv2.Rodrigues(r)[0],t

def angle(R):return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
def rms(a,b):return float(np.sqrt(np.mean(np.sum((a.reshape(-1,2)-b.reshape(-1,2))**2,axis=1))))
for o in obs:
 assert o['metadata']['color_intrinsics']==i
 ids=sorted(set(o['l515']['ids'])&set(o['reachy']['ids']))
 obj=b.getChessboardCorners()[ids].astype(np.float32)
 def points(key):
  d=dict(zip(o[key]['ids'],o[key]['corners']))
  return np.array([d[n] for n in ids],np.float32).reshape(-1,1,2)
 a,c=points('l515'),points('reachy')
 objects.append(obj);p1.append(a);p2.append(c)
 R1,t1=pose(obj,a,K1,D1);R2,t2=pose(obj,c,K2,D2)
 R=R2@R1.T;t=t2-R@t1
 independent.append((R,t))
def fit(keep):
 return cv2.stereoCalibrate([objects[n] for n in keep],[p1[n] for n in keep],[p2[n] for n in keep],K1.copy(),D1.copy(),K2.copy(),D2.copy(),(640,480),flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,100,1e-9))
f=fit(list(range(len(obs))));R,T=f[5],f[6]
held=[]
for n in range(len(obs)):
 q=fit([j for j in range(len(obs)) if j!=n]);r,t=q[5],q[6]
 r1,t1=pose(objects[n],p1[n],K1,D1)
 pred,_=cv2.projectPoints(objects[n],cv2.Rodrigues(r@r1)[0],r@t1+t,K2,D2)
 r2,t2=pose(objects[n],p2[n],K2,D2)
 pred1,_=cv2.projectPoints(objects[n],cv2.Rodrigues(r.T@r2)[0],r.T@(t2-t),K1,D1)
 held.append(dict(view=paths[n].parent.name,reachy_prediction_rms_px=rms(pred,p2[n]),l515_prediction_rms_px=rms(pred1,p1[n]),translation_delta_mm=float(np.linalg.norm(t-T)*1000),rotation_delta_deg=angle(r@R.T)))
E=obs[0]['metadata']['depth_to_color'];Rd=np.array(E['rotation_column_major']).reshape(3,3,order='F');td=np.array(E['translation_m']).reshape(3,1)
def transform(r,t):
 x=np.eye(4);x[:3,:3]=r;x[:3,3]=t.ravel();return x.tolist()
report=dict(status='candidate_not_deployed',l515_serial=obs[0]['metadata']['serial'],views=len(obs),stereo_rms_pixels=f[0],
 convention='p_reachy_optical = R @ p_l515_optical + t; metres; x right, y down, z forward',
 l515_color_to_reachy_optical=transform(R,T),l515_depth_to_reachy_optical=transform(R@Rd,R@td+T),
 translation_norm_m=float(np.linalg.norm(T)),square_length_m=.0235,scale_basis='estimated iPad square size; translation scale is unvalidated',
 scope='fixed head and sensor mounting only; no base_link transform or moving-head calibration',
 reachy_intrinsics=candidate,l515_color_intrinsics=i,
 leave_one_pair_out=held,independent_pair_consistency=[dict(translation_delta_mm=float(np.linalg.norm(t-T)*1000),rotation_delta_deg=angle(r@R.T)) for r,t in independent],
 head_poses=[dict(view=p.parent.name,before=o.get('head_pose_before'),after=o.get('head_pose_after'),body_yaw=o.get('body_yaw_before')) for p,o in zip(paths,obs)],
 validation_note='Fixed intrinsics; each held-out board pose is fitted in one camera, then projected into the other. Factory calibration and estimated board scale are not independent metric ground truth.')
(root/'paired-extrinsics-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['reachy_intrinsics','head_poses','l515_color_intrinsics']},indent=2))
