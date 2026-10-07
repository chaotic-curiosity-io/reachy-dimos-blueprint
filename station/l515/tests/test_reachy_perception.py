import json
import time
import numpy as np
from station.l515.reachy_perception import open_memory, remember, recall


def test_memory_filters_and_retains_camera_only_observations(tmp_path):
    path=tmp_path/'memory.sqlite3'
    db=open_memory(path)
    for i,name in enumerate(['cup','book','cup']):
        remember(db,dict(observation_id=str(i),source_received_at_unix=float(i),
                         coordinate_space='image_pixels',position_3d=None,
                         objects=[dict(name=name,confidence=.8,bbox_xyxy=[1,2,3,4])]))
    db.close()
    results=recall(path,'CUP',1)
    assert len(results)==1 and results[0]['observation_id']=='2'
    assert results[0]['position_3d'] is None
    assert recall(path,'person')==[]
    assert recall(tmp_path/'absent')==[]


def test_native_detection_message_roundtrip():
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
    from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
    from ultralytics.engine.results import Results
    import torch
    image=Image.from_opencv(np.zeros((100,100,3),dtype=np.uint8),ts=time.time(),frame_id='reachy_head_camera_optical')
    raw=Results(image.to_opencv(),path='',names={0:'person'},boxes=torch.tensor([[10,20,50,80,.9,0]]))
    result=ImageDetections2D.from_ultralytics_result(image,[raw])
    message=result.to_ros_detection2d_array()
    message.header.frame_id=image.frame_id
    for detection in message.detections:
        detection.header.frame_id=image.frame_id
    decoded=Detection2DArray.lcm_decode(message.lcm_encode())
    assert len(decoded.detections)==1
    assert decoded.header.frame_id==image.frame_id
    assert result[0].name=='person'
    assert result.annotated_image().to_opencv().sum()>0
    empty=ImageDetections2D(image,[]).to_ros_detection2d_array()
    assert len(Detection2DArray.lcm_decode(empty.lcm_encode()).detections)==0
