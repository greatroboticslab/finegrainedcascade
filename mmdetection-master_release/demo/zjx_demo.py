from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pycocotools import coco
config = '/home/mtsu/workspace/zjx/mmdetection-master/configs/laser/cascade_rcnn_r50_fpn_20e_coco.py'
checkpoint = '/home/mtsu/workspace/zjx/mmdetection-master/work_dirs/cascade_rcnn_r50_fpn_20e_coco/latest.pth'
model = init_detector(config, checkpoint, device='cuda:0')
val_file = '/data/laser_detection/red/labelme2coco/val.json'
coco = coco.COCO(val_file)
imgIds = coco.getImgIds()
val_list = coco.loadImgs(imgIds)
for v in val_list:
    #img = '/data/laser_detection/red/red-laser/IMG350.jpg' #'demo.jpg'
    print(v)
    img = '/data/laser_detection/red/red-laser/' + v['file_name']
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result, score_thr=0.3)

