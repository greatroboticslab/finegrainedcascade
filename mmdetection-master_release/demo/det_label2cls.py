import os
from pycocotools import coco
js_file = '/data/laser_detection/red/labelme2coco/val.json'
coco = coco.COCO(js_file)
imgIds = coco.getImgIds()
img_list = coco.loadImgs(imgIds)
txt_list = []
cls2clsid = {0:2, 2:4, 4:3, 6:5, 9:1}
print(coco.cats)
for i in img_list:
    #img = '/data/laser_detection/red/red-laser/IMG350.jpg' #'demo.jpg'
    ann_list = coco.loadAnns(coco.getAnnIds(i['id']))
    for a in ann_list:
        if a['category_id'] in cls2clsid:
            txt_list.append('%s %d\n' % (os.path.basename(i['file_name']),
                                         cls2clsid[a['category_id']] - 1))
exit(0)
with open('/data/laser_detection/red/cls_label/val.txt', 'w') as f:
    f.writelines(txt_list)
