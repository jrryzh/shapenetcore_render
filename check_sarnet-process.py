import os
import sys

dataset_dir = "/home/add_disk1/zhangjinyu/shapenetcorev2_sarnet_output/"
for category in os.listdir(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, category)):
        model_lst = os.listdir(os.path.join(dataset_dir, category))
        print(f"{category}: {len(model_lst)}")