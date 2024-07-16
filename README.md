当前pipeline：
1. 从 /home/add_disk_e/objaverse/hf-objaverse-v1 获取全部模型
<!-- 2. glb2obj 存在 /home/add_disk_e/objaverse_lvis_trimesh_objs/
3. normobj -->
2-3.trimesh_multithread_convert
4. obj2urdf
5. render/sapien_filter.py
当前存储路径顺序：
1. objaverse_lvis_trimesh_normalized_objs
2. objaverse_lvis_trimesh_normalized_objs_output
3. objaverse_lvis_trimesh_normalized_objs_output_sarnet + objaverse_lvis_trimesh_normalized_objs_output_template_FPS


######################################################

模型位置：
/home/add_disk_e/ShapeNetCore.v2
标注文件：
/home/add_disk_e/ShapeNetCore.v2/taxonomy.json 
渲染结果位置：
/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_render_output
sarnet点云位置：
/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_sarnet_output

步骤：
conda activate objaverse
python /home/fudan248/zhangjinyu/code_repo/shapenetcorev2/obj2urdf.py
python /home/fudan248/zhangjinyu/code_repo/shapenetcorev2/render_all.py

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
