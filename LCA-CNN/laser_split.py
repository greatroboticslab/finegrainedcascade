import os
import shutil

train_dict = {}
with open('/data/laser_detection/red/cls_label/train.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        img, label = l.split(' ')
        train_dict[img.split('.')[0]] = int(label)
#print(train_dict)
val_dict = {}
with open('/data/laser_detection/red/cls_label/val.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        img, label = l.split(' ')
        val_dict[img.split('.')[0]] = int(label)
#print(train_dict)

src_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/crop_bbox_new/'
imgs = os.listdir(src_dir)
for i in imgs:
    i_name = i.split('.')[0]
    if i_name in train_dict:
        dst_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/data_split/train_new/'
        cls = train_dict[i_name]
        dst_folder = dst_dir + '%d/' % cls
        shutil.copyfile(src_dir + i, dst_folder + i)
    if i_name in val_dict:
        dst_dir = '/data/laser_detection/work_dirs/RED_cascade_rcnn_r50_fpn_20e_coco/data_split/val_new/'
        cls = val_dict[i_name]
        dst_folder = dst_dir + '%d/' % cls
        shutil.copyfile(src_dir + i, dst_folder + i)


