import os
import sys

if __name__ == '__main__':
    sarnet_data_root = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_output"
    
    category_list = os.listdir(sarnet_data_root)
    for category in category_list:
        if category == "template_FPS" or category == "train.json":
            continue
        
        model_list = os.listdir(os.path.join(sarnet_data_root, category))
        for model in model_list:
            data_list = os.listdir(os.path.join(sarnet_data_root, category, model))
            if len(data_list) < 1800:
                # print("Warning: {}/{} has less than 1800 data".format(category, model))
                print("{}/{}/{}".format(sarnet_data_root, category, model))
                # os.system("rm -rf {}/{}/{}".format(sarnet_data_root, category, model))
    