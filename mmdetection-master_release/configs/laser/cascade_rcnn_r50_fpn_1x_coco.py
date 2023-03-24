_base_ = [
    'cascade_rcnn_r50_fpn.py',
    'coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
load_from = '/home/mtsu/workspace/zjx/mmdetection-master/configs/laser/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'
