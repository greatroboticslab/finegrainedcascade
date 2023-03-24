import os
import shutil
'''
js_files = os.listdir('/data/laser_detection/red/old_label/')
for js in js_files:
    file_data = ""
    with open('/data/laser_detection/red/old_label/'+ js, 'r') as f:
        for l in f:
            l_new = l.replace("red laser day 3", "day3 red laser")
            file_data += l_new
    with open('/data/laser_detection/red/label/'+ js, 'w') as f:
        f.write(file_data)
'''
js_files = os.listdir('/data/laser_detection/green/label/')
for js in js_files:
    file_data = ""
    with open('/data/laser_detection/green/label/'+ js, 'r') as f:
        for l in f:
            l_new = l
            if '"green laser"' in l_new:
                for i in ['day1', 'day2', 'day3', 'day4']:
                    if js in os.listdir('/data/laser_detection/green/%s/label/' % i):
                        l_new = l_new.replace('green laser"', '%s green laser"' % i)
                #print(l_new)
            if 'imagePath' in l:
                l_new = l_new.replace('"imagePath": "', '"imagePath": "../green-laser/')
                l_new = l_new.replace('../green-laser/../green-laser/', '../green-laser/')
                l_new = l_new.replace('../green-laser/../', '../green-laser/')
                l_new = l_new.replace('../green-laser/green-laser/', '../green-laser/')
            file_data += l_new
    with open('/data/laser_detection/green/cleaned_label/'+ js, 'w') as f:
        f.write(file_data)