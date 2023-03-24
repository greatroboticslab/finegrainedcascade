import os

import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pycocotools import coco
config = '/home/mtsu/workspace/zjx/mmdetection-master/configs/laser/cascade_rcnn_r50_fpn_20e_coco.py'
checkpoint = '/home/mtsu/workspace/zjx/mmdetection-master/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/latest.pth'
model = init_detector(config, checkpoint, device='cuda:0')
val_file = '/data/laser_detection/red/labelme2coco/val.json' #train.json'
coco = coco.COCO(val_file)
imgIds = coco.getImgIds()
val_list = coco.loadImgs(imgIds)
for v in val_list:
    #img_filename = '/data/laser_detection/red/red-laser/IMG350.jpg' #'demo.jpg'
    #print(v)
    img_filename = '/data/laser_detection/red/red-laser/' + v['file_name']
    img_content = mmcv.bgr2rgb(mmcv.imread(img_filename))
    #print(img_content.shape)
    result = inference_detector(model, img_filename)

    score_thr = 0 # 0.3
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    #print(bboxes.shape, labels.shape)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        cls_prediction = model.CLASSES[label]
        cropped_img = np.ascontiguousarray(img_content)[bbox_int[1]:bbox_int[3],
                      bbox_int[0]:bbox_int[2]]
        plt.imshow(cropped_img)
        plt.title(str(os.path.basename(img_filename) + ': ' + cls_prediction))
        #plt.show()
        crop_save_dst = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/crop_bbox_new/' + os.path.basename(img_filename)
        if os.path.isfile(crop_save_dst):
            print(crop_save_dst, ' exists!')
            plt.imsave(crop_save_dst + '.png', cropped_img)
        else:
            plt.imsave(crop_save_dst, cropped_img)

    #show_result_pyplot(model, img_filename, result, score_thr=0.3)
