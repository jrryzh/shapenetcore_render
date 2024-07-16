import numpy as np

def load_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def save_obj(vertices, file_path):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def load_pose(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            matrix.append([float(x) for x in line.strip().split()])
    return np.array(matrix)

def apply_transformation(vertices, transformation_matrix):
    num_vertices = vertices.shape[0]
    homogenous_vertices = np.hstack([vertices, np.ones((num_vertices, 1))])
    transformed_vertices = homogenous_vertices.dot(transformation_matrix.T)
    return transformed_vertices[:, :3]

categories = {
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

# /home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped_render_output/airplane/3b0efeb0891a9686ca9f0727e23831d9/0018_rgb.png
# Load obj file
vertices = load_obj('/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped/03325088/1c684a5a113c4536b7a867e9b35a1295/models/model_normalized.obj')

# Load pose file
transformation_matrix = load_pose('/home/fudan248/zhangjinyu/tmp/test0709/0260_pose.txt')

# Apply transformation
transformed_vertices = apply_transformation(vertices, transformation_matrix)

# Save new obj file
save_obj(transformed_vertices, '/home/fudan248/zhangjinyu/tmp/test0709/0260_transposed.obj')
