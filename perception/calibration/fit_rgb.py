"""Fit saved ChArUco observations; evaluate leave-one-view-out prediction."""
from pathlib import Path
import json
import os
import cv2
import numpy as np
# Data/output folder: next to this script by default; override with CALIBRATION_DIR.
root=Path(os.environ.get('CALIBRATION_DIR') or Path(__file__).parent)
folder=root/'rgb-ipad'
board=cv2.aruco.CharucoBoard((5,7),.0235,.01175,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
files=sorted(folder.glob('*.json'))
obs=[json.loads(f.read_text()) for f in files]
obj=[board.getChessboardCorners()[np.array(o['corner_ids'])].astype(np.float32) for o in obs]
img=[np.array(o['corners'],np.float32).reshape(-1,1,2) for o in obs]
assert all(o['image_size']==[1280,720] for o in obs)
size=(1280,720)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,100,1e-9)
results={}
for name,flags in [('brown_5',0),('brown_4_fixed_k3',cv2.CALIB_FIX_K3)]:
 fit=cv2.calibrateCameraExtended(obj,img,size,None,None,flags=flags,criteria=criteria)
 rms,K,D,rvecs,tvecs,stdI,stdE,per=fit
 held=[];intrinsics=[]
 for i in range(len(obj)):
  keep=[j for j in range(len(obj)) if j!=i]
  _,k,d,_,_=cv2.calibrateCamera([obj[j] for j in keep],[img[j] for j in keep],size,None,None,flags=flags,criteria=criteria)
  ok,r,t=cv2.solvePnP(obj[i],img[i],k,d)
  if not ok:raise RuntimeError('held-out pose fit failed')
  p,_=cv2.projectPoints(obj[i],r,t,k,d)
  held.append(float(np.sqrt(np.mean(np.sum((p-img[i])**2,axis=2)))))
  intrinsics.append([k[0,0],k[1,1],k[0,2],k[1,2]])
 result=dict(rms_pixels=rms,K=K.tolist(),D=D.flatten().tolist(),intrinsic_standard_deviations=stdI.flatten().tolist(),per_view_rms_pixels=per.flatten().tolist(),leave_one_view_out_rms_pixels=held,leave_one_view_out_intrinsics=intrinsics,leave_one_view_out_intrinsic_range=np.ptp(intrinsics,axis=0).tolist())
 results[name]=result
 print(name,json.dumps(result),flush=True)
allpoints=np.concatenate(img).reshape(-1,2)
hull=cv2.convexHull(allpoints)
report=dict(image_size=list(size),views=len(obs),source_files=[f.name for f in files],square_length_m=.0235,scale_basis='estimated, not measured',models=results,corner_bounds_pixels=[allpoints.min(axis=0).tolist(),allpoints.max(axis=0).tolist()],corner_convex_hull_image_fraction=float(cv2.contourArea(hull)/(1280*720)),validation_note='Leave-one-view-out refits intrinsics without that view; held-out board pose is estimated from its corners. This is not an independent 3D ground-truth test.',deployment_status='candidate only; not applied')
(root/'rgb-fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
canvas=np.full((720,1280,3),245,np.uint8)
for i,pts in enumerate(img):
 color=tuple(int(x) for x in np.random.default_rng(i).integers(30,210,3))
 for x,y in pts.reshape(-1,2):cv2.circle(canvas,(round(float(x)),round(float(y))),4,color,-1)
cv2.imwrite(str(root/'corner-coverage.jpg'),canvas)
best=min(results,key=lambda n:np.mean(results[n]['leave_one_view_out_rms_pixels']))
K=np.array(results[best]['K']);D=np.array(results[best]['D'])
source=cv2.imread(str(files[0].with_suffix('.jpg')))
corrected=cv2.undistort(source,K,D,None,K)
cv2.putText(source,'Original',(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.putText(corrected,'Candidate correction',(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.imwrite(str(root/'rgb-correction-comparison.jpg'),np.hstack([source,corrected]))
print('Lowest mean held-out error model:',best)
