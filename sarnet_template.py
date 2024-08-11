import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d


def load_obj(path_to_file):
    vertices = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
    vertices = np.asarray(vertices)
    return vertices

def save_to_obj_pts(verts, path):
    with open(path, 'w') as file:
        for v in verts:
            file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

# def farthest_point_sampling(points, sample_num):
#     N, D = points.shape
#     centroids = np.zeros((sample_num,))
#     distance = np.ones((N,)) * 1e10
#     farthest = random.randint(0, N-1)
#     for i in range(sample_num):
#         centroids[i] = farthest
#         centroid = points[farthest, :]
#         dist = np.sum((points - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance)
#     return points[centroids.astype(np.int32)]

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    global centroid
    global scale
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale
    return pc, centroid, scale

def save_to_obj_pts(verts, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as file:
        for v in verts:
            file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C

def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts

def process_category(cate_id, cate_name, source_folder, target_folder, sample_num):
    template_folder_lst = os.listdir(os.path.join(source_folder, cate_id))
    template_folder = template_folder_lst[random.randint(0, len(template_folder_lst) - 1)]
    source_path = os.path.join(source_folder, cate_id, template_folder, "models/model_normalized.obj")
    target_path = os.path.join(target_folder, f"{cate_name}_fps_{sample_num}_normalized.obj")
    # Load OBJ file
    vertices = load_obj(source_path)
    print(f"Loading template for category: {cate_name}")
    print(f"Vertices shape: {vertices.shape}")
    # Perform FPS sampling on vertices
    selected_pts = farthest_point_sampling(vertices, sample_num)
    sampled_vertices = vertices[selected_pts]
    # Normalize
    sampled_vertices, _, _ = pc_normalize(sampled_vertices)
    # Save the sampled vertices to a new OBJ file
    save_to_obj_pts(sampled_vertices, target_path)
    print(f"Finish processing template for category: {cate_name}")
    
def process_category_test(cate_id, cate_name, source_folder, target_folder, sample_num):
    template_folder_lst = os.listdir(os.path.join(source_folder, cate_id))
    template_folder = template_folder_lst[random.randint(0, len(template_folder_lst) - 1)]
    source_path = os.path.join(source_folder, cate_id, template_folder, "models/model_normalized.obj")
    target_path = os.path.join(target_folder, f"{cate_name}_fps_{sample_num}_normalized.obj")
    # Load OBJ file
    mesh = o3d.io.read_triangle_mesh(source_path)
    # vertices = load_obj(source_path)
    print(f"Loading template for category: {cate_name}")
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd = mesh.sample_points_poisson_disk(number_of_points=2000, pcl=pcd)
    vertices = np.asarray(pcd.points)
    # Perform FPS sampling on vertices
    selected_pts = farthest_point_sampling(vertices, sample_num)
    sampled_vertices = vertices[selected_pts]
    # Normalize
    sampled_vertices, _, _ = pc_normalize(sampled_vertices)
    # Save the sampled vertices to a new OBJ file
    save_to_obj_pts(sampled_vertices, target_path)
    print(f"Finish processing template for category: {cate_name}")

# if __name__ == '__main__':
#     # 应用
#     random.seed(42)

#     source_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped"
#     target_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_fps"
#     sample_num = 128  # 采样点数量

#     categories = {
#         "02691156": "airplane",
#         "02828884": "bench",
#         "02876657": "bottle",
#         "02958343": "car",
#         "03001627": "chair",
#         "03211117": "display",
#         "03261776": "earphone",
#         "03325088": "faucet",
#         "03467517": "guitar",
#         "03624134": "knife",
#         "03636649": "lamp",
#         "03642806": "laptop",
#         "03691459": "loudspeaker",
#         "03790512": "motorcycle",
#         "03928116": "piano",
#         "04004475": "printer",
#         "04074963": "remote",
#         "04256520": "sofa",
#         "04379243": "table",
#         "02818832": "bed",
#     }

#     for cate_id, cate_name in categories.items():
#         template_folder_lst = os.listdir(os.path.join(source_folder, cate_id))
#         template_folder = template_folder_lst[random.randint(0, len(template_folder_lst)-1)]
#         source_path = os.path.join(source_folder, cate_id, template_folder, "models/model_normalized.obj")
#         target_path = os.path.join(target_folder, f"{cate_name}_fps_128_normalized.obj")
#         # 加载OBJ文件
#         vertices = load_obj(source_path)
#         print("Loading template for category: ", cate_name)
#         print("Vertices shape: ", vertices.shape)
#         # 对顶点进行FPS采样
#         selected_pts = farthest_point_sampling(vertices, sample_num)
#         sampled_vertices = vertices[selected_pts]
#         # 归一化
#         sampled_vertices, _, _ = pc_normalize(sampled_vertices)
#         # 保存采样后的顶点为新的OBJ文件
#         save_to_obj_pts(sampled_vertices, target_path)
#         print("Finish processing template for category: ", cate_name)

if __name__ == '__main__':
    random.seed(42)

#   source_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped"
    source_folder = "/mnt/test/data/shapenet/flipped"
#   target_folder = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_fps"
    target_folder = "/mnt/test/data/shapenet/shapenetcorev2_sarnet_fps"
    sample_num = 36  # Number of sample points

    categories_old = {
        "02691156": "airplane",
        "02828884": "bench",
        "02876657": "bottle",
        "02958343": "car",
        "03001627": "chair",
        "03211117": "display",
        "03261776": "earphone",
        "03325088": "faucet",
        "03467517": "guitar",
        "03624134": "knife",
        "03636649": "lamp",
        "03642806": "laptop",
        "03691459": "loudspeaker",
        "03790512": "motorcycle",
        "03928116": "piano",
        "04004475": "printer",
        "04074963": "remote",
        "04256520": "sofa",
        "04379243": "table",
        "02818832": "bed",
    }

    categories = {
    '04460130': 'tower', 
    '04468005': 'train', 
    '04530566': 'vessel', 
    '04554684': 'washer', 
    }

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_category_test, cate_id, cate_name, source_folder, target_folder, sample_num)
            for cate_id, cate_name in categories.items()
        ]

        # Optionally wait for all futures to complete and check for exceptions
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")