import os
import shutil


src_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/all_crop_bbox/data_112/'#data/'
dst_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/all_crop_box_by_name_112/'
imgs = os.listdir(src_dir)
for i in imgs:
    shutil.copyfile(src_dir + i, dst_dir + str(i.split('_')[1]) + '/' + i)
