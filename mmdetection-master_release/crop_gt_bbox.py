import os

import mmcv

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pycocotools import coco
import matplotlib.pyplot as plt

config = '/home/mtsu/workspace/zjx/mmdetection-master/configs/laser/cascade_rcnn_r50_fpn_20e_coco.py'
checkpoint = '/home/mtsu/workspace/zjx/mmdetection-master/work_dirs/cascade_rcnn_r50_fpn_20e_coco/latest.pth'
#model = init_detector(config, checkpoint, device='cuda:0')
val_file = '/data/laser_detection/red/labelme2coco/train.json'#val.json'
coco = coco.COCO(val_file)
imgIds = coco.getImgIds()
val_list = coco.loadImgs(imgIds)
bbox = coco.loadAnns(coco.getAnnIds())
for b in bbox:
    img_path = '/data/laser_detection/red/red-laser/' + coco.loadImgs(b['image_id'])[0]['file_name']
    x, y, w, h = b['bbox']
    class_name = coco.loadCats(b['category_id'])[0]['name']
    if class_name not in ('day1 red laser', 'day2 red laser', 'day3 red laser', 'red laser day 4', 'red laser day 5'):
        continue
    #print(class_name)
    cropped_img = mmcv.bgr2rgb(mmcv.imread(img_path))[y:y + h, x:x + w] #plt.imread(img_path)[y:y + h, x:x + w]
    plt.imshow(cropped_img)
    plt.title(str(os.path.basename(img_path) + ': ' + class_name))
    #plt.show();continue
    crop_save_dst = '/data/laser_detection/red//crop_gt_new_codec/' + os.path.basename(img_path)
    if os.path.isfile(crop_save_dst):
        print(crop_save_dst, ' exists!')
        plt.imsave(crop_save_dst + '.png', cropped_img)
    else:
        plt.imsave(crop_save_dst, cropped_img)
