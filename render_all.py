from multiprocessing import Process
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def run_script(synset_id):
    os.system(f"python /home/fudan248/zhangjinyu/code_repo/shapenetcorev2/render_flip.py --synset_id {synset_id}")

# 定义不同的 synset_id
#categories = {
    # "02691156": "airplane",
    # "02828884": "bench",
    # "02876657": "bottle",
    # "02958343": "car",
    # "03001627": "chair",
    # "03211117": "display",
    # "03261776": "earphone",
    # "03325088": "faucet",
    # "03467517": "guitar",
    # "03624134": "knife",
    # "03636649": "lamp",
    # "03642806": "laptop",
    # "03691459": "loudspeaker",
    # "03790512": "motorcycle",
    # "03928116": "piano",
    # "04004475": "printer",
    # "04074963": "remote",
    # "04256520": "sofa",
    # "04379243": "table",
    # "02818832": "bed",
#}
from commons import categories

# 创建进程
processes = [Process(target=run_script, args=(synset_id,)) for synset_id in list(categories.keys())[50:]]

# 启动所有进程
for p in processes:
    p.start()
t = time.time()

# 等待所有进程完成
for p in processes:
    p.join()

print(f"All processes done in {time.time() - t:.2f} seconds")