import os
import threading

"""
Simple script to wrap an .obj file into an .urdf file, modified to use multithreading for processing multiple files concurrently.
"""

def convert_obj_coordinates(input_file, output_file):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        with open(output_file, 'w') as file:
            for line in lines:
                if line.startswith('v '):
                    parts = line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    # 将顶点坐标从 (x, y, z) 转换为 (x, z, -y)
                    new_line = f"v {x} {-z} {y}\n"
                    file.write(new_line)
                else:
                    file.write(line)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_category(category_path, shapenet_datadir, flip_shapenet_datadir):
    for obj_id in os.listdir(category_path):
        obj_path = os.path.join(category_path, obj_id, "models", "model_normalized.obj")
        flip_obj_path = obj_path.replace(shapenet_datadir, flip_shapenet_datadir)
        os.makedirs(os.path.dirname(flip_obj_path), exist_ok=True)
        convert_obj_coordinates(obj_path, flip_obj_path)

if __name__ == "__main__":
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
    
    shapenet_datadir = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/ShapeNetCore.v2/"
    flip_shapenet_datadir = "/home/add_disk/zhangjinyu/shapenetcorev2_flipped/"
    threads = []
    
    for synsetId in categories.keys():
        category_path = os.path.join(shapenet_datadir, synsetId)
        
        if not os.path.exists(category_path):
            print("Category path does not exist: ", category_path)
            continue
        
        thread = threading.Thread(target=process_category, args=(category_path, shapenet_datadir, flip_shapenet_datadir))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
