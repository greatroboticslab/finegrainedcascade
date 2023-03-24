# The new config inherits a base config to highlight the necessary modification
_base_ = [
    'faster_rcnn_r50_fpn.py',
    'coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
'''
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('day1 red laser', 'day2 green laser',
           'green laser day 3', 'red laser day 4', 'red laser day 5')
data = dict(
    train=dict(
        img_prefix='/data/laser_detection/red/labelme2coco/',
        classes=classes,
        ann_file='/data/laser_detection/red/labelme2coco/train.json'),
    val=dict(
        img_prefix='/data/laser_detection/red/labelme2coco/',
        classes=classes,
        ann_file='/data/laser_detection/red/labelme2coco/val.json'),
    test=dict(
        img_prefix='/data/zhong_data/redlaser/labelme2coco/',
        classes=classes,
        ann_file='/data/zhong_data/redlaser/labelme2coco/val.json'))
'''
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/mtsu/workspace/zjx/mmdetection-master/configs/laser/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'