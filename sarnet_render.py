import numpy as np
import cv2
import os
import multiprocessing
from multiprocessing import Pool

from glob import glob
from copy import deepcopy
from transforms3d.euler import euler2mat
import math

import shutil

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    global centroid
    global scale
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale
    return pc, centroid, scale

def load_obj(path_to_file):
    """ Load obj file.

    Args:
    path_to_file: path

    Returns:
    vertices: ndarray
    
    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            else:
                continue
    vertices = np.asarray(vertices)
    
    return vertices

def save_to_obj_pts(verts, path):
    with open(path, 'w') as file:
        for v in verts:
            file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def create_folders(path):
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i+n] for i in range(0, len(arr), n)]

def count_vertices_in_obj(dir, file_path):
    num_vertices = 0
    with open(os.path.join(dir, file_path), 'r') as file:
        for line in file:
            if line.startswith('v '):  # 每个顶点行以 'v ' 开头
                num_vertices += 1
    return num_vertices
        
def check_obj_complete(dir):
    file_list = os.listdir(dir)
    obj_lst = [f for f in file_list if f.endswith('.obj')]
    for obj_file in obj_lst:    
        num_vertices = count_vertices_in_obj(dir, obj_file)
        if num_vertices != 4096:
            return False
    return True

# 0:reflectional, 1:rottional

# categories2sym = {
#         "Bottle":1, "Box":0, "Bucket":1, "Cart":0, "CoffeeMachine":0, "Dispenser":1,
#         "Door":0, "KitchenPot":1, "Microwave":0, "Oven":0, "Refrigerator":0, "Remote":0,
#         "Safe":0, "StorageFurniture":0, "Suitcase":0, "Table":0, "TrashCan":1, "WashingMachine":0,
#         "Camera":0, "Chair":0, "Clock":0, "Dishwasher":0, "Display":0, "Eyeglasses":0,
#         "Fan":0, "Faucet":0, "FoldingChair":0, "Globe":0, "Kettle":0, "Keyboard":0, 
#         "Knife":0, "Lamp":0, "Laptop":0, "Lighter":0, "Mouse":0, "Pen":1, 
#         "Phone":0, "Pliers":0, "Printer":0, "Scissors":0, "Stapler":0, "Switch":0,
#         "Toaster":0, "Toilet":0, "USB":0, "Window":0, "Packaging":0, "Sponge":0, "Banana":2
#     }
# categories2sym = {
#         "balloon":1,
#         # Cups and containers
#         "cup":1, "mug":0, "bottle":1, "beer_bottle":1, "jar":1, "can":1, "water_bottle":1,
#         # Kitchen utensils
#         "fork":0, "spoon":0, "knife":0, "plate":1, "bowl":1,
#         # Kitchen appliances
#         "toaster":1, "kettle":0, "wineglass":1,
#         # Toys
#         "ball":1,
#         # Fruits
#         "apple":1, "banana":0, "orange":0
#     }

#categories2sym = {
    #'airplane':2, 'bench':2, 'bottle':1, 'car':2, 'chair':2, 'display':2, 'earphone':2, 'faucet':2, 'guitar':2, 'knife':2, 'lamp':2, 'laptop':2, 'loudspeaker':2, 'motorcycle':2, 'piano':2, 'printer':2, 'remote':2, 'sofa':2, 'table':2, 'bed':2
#}
# categories2sym = {'earphone':2, 'remote':2, 'bed':2}
#categories2sym = {'airplane':2, 'ashcan':1, 'bag':1, 'basket':1, 'bathtub':1, 'bed':2, 'bench':2, 'bicycle':2, 'birdhouse':2, 'bookshelf':2, 'bottle':1, 'bowl':1, 'bus':2, 'cabinet':1, 'camera':2, 'can':1, 'cap':1, 'car':2, 'chair':2, 'clock':2, 'computer':2, 'cup':1, 'curtain':1, 'desk':1, 'door':2, 'dresser':1, 'flower_pot':1, 'glass_box':1, 'guitar':2, 'keyboard':2, 'lamp':2, 'laptop':2,'mantel':1,'monitor':2, 'night_stand':1, 'person':2, 'piano':2, 'plant':1}
#categories2sym = {'airplane':2, 'ashcan':1, 'bag':1, 'basket':1, 'bathtub':1, 'bed':2, 'bench':2, 'birdhouse':2, 'bookshelf':2, 'bottle':1,} 
categories2sym = {'laptop': 2, 'loudspeaker': 2, 'mailbox': 2, 'microphone': 2, 'microwave': 2, 'motorcycle': 2, 'mug': 2, 'piano': 2, 'pillow': 2, 'pistol': 2, 'pot': 1, 'printer': 2, 'remote': 2, 'rifle': 2, 'rocket': 2, 'skateboard': 2, 'sofa': 2, 'stove': 2, 'table': 2, 'telephone': 2, 'cellphone': 2, 'tower': 2, 'train': 2, 'vessel': 2, 'washer': 2}
VIEW=300
SAMPLE_NUM=4096
#source_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped_render_output/"
source_folder = "/mnt/test/data/shapenet/shapenetcorev2_render_output2/"
target_folder = "/mnt/test/data/shapenet/shapenetcorev2_sarnet_output/"
# target_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_output/"
#target_template_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_fps"
target_template_folder = "/mnt/test/data/shapenet/shapenetcorev2_sarnet_fps/"
def process_template():
    create_folders(target_template_folder)
    for cate_name in categories2sym.keys():
        template_path = os.path.join(source_folder, cate_name, 'template_128.obj')
        template = load_obj(template_path)
        
        # if cate_name not in ['Packaging', 'Sponge', 'Banana']: #if it is sapien model
        #sapien load model:y-up, z_forward -> z-up, -x forward,  
        rot_axis = euler2mat(np.pi/2., 0, -np.pi/2, axes='sxyz')
        template = np.dot(template, rot_axis.T)
        template, _, _ = pc_normalize(template)

        save_path = os.path.join(target_template_folder, '{}_fps_128_normalized.obj'.format(cate_name))
        save_to_obj_pts(template, save_path)

def get_all_instace_path_list():
    all_instance_path_list = []
    for cate_name in categories2sym:
        instance_name_list = os.listdir(os.path.join(source_folder, cate_name))
        
        # 截断
        if len(instance_name_list) > 100:
            instance_name_list = instance_name_list[:100]
        
        for instance_name in instance_name_list:
            if instance_name == 'template_128.obj' or instance_name == 'intrinsic.txt':
                continue
            instance_path = os.path.join(source_folder, cate_name, instance_name)
            all_instance_path_list.append(instance_path)
        
        # DEBUG
        # break

    return all_instance_path_list

# processing each category
def _processing(instance_path_list):
    for instance_path in instance_path_list:
        try:
            print("processing: ", instance_path)
            target_path = instance_path.replace(source_folder, target_folder)
            cate_name = instance_path.split('/')[-2]
            
            # step1: check if the instance has enough views
            if len(os.listdir(instance_path)) < 1500:
                print("not enough views: ", instance_path)
                # shutil.rmtree(target_path)
                continue
            
            # step2: create folders and save files
            create_folders(target_path)
            
            if len(os.listdir(target_path)) >= 1800:
                print("already processed: ", target_path)
                continue
            
            for i in range(0, VIEW):
                index = str(i).zfill(4)

                pcd_path = os.path.join(instance_path, '{}_pcd.obj'.format(index))
                pose_path = os.path.join(instance_path, '{}_pose.txt'.format(index))
                size_path = os.path.join(instance_path, 'scale.txt')
                # template_path = os.path.join(instance_path, 'template_128.obj')

                # load and norm the pcd
                _obsv_pcd = load_obj(pcd_path)
                pts_num = _obsv_pcd.shape[0]
                if pts_num < SAMPLE_NUM:
                    choose = np.random.choice(np.arange(pts_num), SAMPLE_NUM, replace=True)
                else:
                    choose = np.random.choice(np.arange(pts_num), SAMPLE_NUM, replace=False)
                _obsv_pcd = _obsv_pcd[choose, :]
                obsv_pcd, centroid, scale = pc_normalize(_obsv_pcd) #(N,3)

                # load pose
                pose = np.loadtxt(pose_path)
                size = np.loadtxt(size_path)
                rot_matrix = pose[:3,:3]
                trans = pose[:3,3] - centroid
                trans_scale = [trans[0], trans[1], trans[2], scale]
                size = size / scale
                trans = trans / scale

                # create the sym, transform to canonical: p' = R.T*p - R.T*t = R.T(p-t)
                cano_pcd = np.dot(obsv_pcd - trans, rot_matrix) 
                
                # DEBUG
                save_cano_path = os.path.join(target_path, '{}_cano.obj'.format(index))
                save_to_obj_pts(cano_pcd, save_cano_path)
                
                # 类别 0（反射对称 y 轴）：
                # 创建 cano_pcd 的深拷贝 cano_sym_pcd。
                # 将 y 坐标乘以 -1，即沿 y 轴反射对称。
                # 类别 1（180度旋转对称）：
                # 创建 cano_pcd 的深拷贝 cano_sym_pcd。
                # 使用 euler2mat 函数将欧拉角 (0, 0, π) 转换为旋转矩阵 rot_axis。
                # 将旋转矩阵应用于 cano_sym_pcd，即绕 z 轴旋转 180 度。
                # 类别 2（反射对称 x 轴）：
                # 创建 cano_pcd 的深拷贝 cano_sym_pcd。
                # 将 x 坐标乘以 -1，即沿 x 轴反射对称。
                if categories2sym[cate_name] == 0:
                    cano_sym_pcd = deepcopy(cano_pcd) #(N,3)
                    cano_sym_pcd[:, 1] *= -1 # y=-y
                elif categories2sym[cate_name] == 1:
                    cano_sym_pcd = deepcopy(cano_pcd)
                    rot_axis = euler2mat(0, 0, np.pi, axes='sxyz')
                    cano_sym_pcd = np.dot(cano_sym_pcd, rot_axis.T)
                elif categories2sym[cate_name] == 2:
                    cano_sym_pcd = deepcopy(cano_pcd) #(N,3)
                    cano_sym_pcd[:, 0] *= -1 # x=-x
                else:
                    print('sym type should be 0, 1')

                # transform to camera frame, p' = R*p+t
                sym_pcd = np.dot(cano_sym_pcd, rot_matrix.T) + trans

                # save results
                save_obsv_path = os.path.join(target_path, '{}_obsv.obj'.format(index))
                save_SC_path = os.path.join(target_path, '{}_SC.obj'.format(index))
                save_rot_path = os.path.join(target_path, '{}_rot.txt'.format(index))
                save_sOC_path = os.path.join(target_path, '{}_sOC.txt'.format(index))
                save_OS_path = os.path.join(target_path, '{}_OS.txt'.format(index))
                                            
                save_to_obj_pts(obsv_pcd, save_obsv_path)
                # save_to_obj_pts(cano_pcd, save_SC_path)     
                save_to_obj_pts(sym_pcd, save_SC_path)                             
                np.savetxt(save_rot_path, rot_matrix)
                np.savetxt(save_sOC_path, trans_scale)
                
                # sapien load model:y-up, z_forward -> z-up, -x forward,  
                # np.savetxt(save_OS_path, [size[2], size[0], size[1]])
                np.savetxt(save_OS_path, [size[0], size[1], size[2]])
        except Exception as e:
            print("Error: ", instance_path)
            print(e)
            #failed_path = "/home/fudan248/zhangjinyu/tmp/test0530/failed_instance_path.txt"
            failed_path = '/mnt/test/data/shapenet/failed_instance_path.txt'
            with open(failed_path, 'a') as file:
                file.write(instance_path)
            continue

# step1: process template
# process_template()

# # step2: process pcd
all_instance_path_list = get_all_instace_path_list()
print(all_instance_path_list)

# import ipdb; ipdb.set_trace()

cpu_num = multiprocessing.cpu_count() //2
split_instance_path_list = chunks(all_instance_path_list, cpu_num)
if cpu_num > len(split_instance_path_list):
    cpu_num = len(split_instance_path_list)

params = []
for i in range(cpu_num):
    params.append(split_instance_path_list[i])

print('using {} CPUs'.format(cpu_num))
with Pool(cpu_num) as p:
    _ = p.map(_processing, params)
p.join()
p.close()
print('finish processing!')


