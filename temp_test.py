# find out the number of rendered files under "/mnt/test/data/shapenet/shapenetcorev2_render_output2"

import os
import shutil

'''path = "/mnt/test/data/shapenet/shapenetcorev2_render_output2"
for category in os.listdir(path):
    count = 0
    for instance in os.listdir(os.path.join(path, category)):
        n = len(os.listdir(os.path.join(path, category, instance)))
        if n > 1400:
            count += 1
        if n == 0:
            os.rmdir(os.path.join(path, category, instance))        
    print(category, count)'''

path_chair = "/mnt/test/data/shapenet/shapenetcorev2_render_output2/car"

remain = 0
count = 0
limit = 100
for instance in os.listdir(path_chair):
    if len(os.listdir(os.path.join(path_chair, instance))) > 1400 and remain < limit:
        remain += 1
        print(instance)
    else :
        try:
            shutil.rmtree(os.path.join(path_chair, instance))
        except OSError as e:
            print("Error: %s : %s" % (os.path.join(path_chair, instance), e.strerror))
    count += 1
    print(count, remain)
