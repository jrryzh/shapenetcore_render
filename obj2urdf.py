# /home/add_disk/zhangjinyu/shapenetcorev2_urdf
import os
import sys

"""
Simple script to wrap an .obj file into an .urdf file.
"""


def split_filename(string):
    path_to_file = os.path.dirname(string)
    filename, extension = os.path.splitext(os.path.basename(string))
    return path_to_file, filename, extension


def check_input():
    if len(sys.argv) == 1:
        print("Provide a <file.obj> as argument")
        sys.exit()
    elif len(sys.argv) > 2:
        print("Too many arguments")
        sys.exit()
    _, _, ext = split_filename(sys.argv[1])
    if ext != ".obj":
        print("Incorrect extension (<{}> instead of <.obj>)".format(ext))
        sys.exit()
    if not os.path.exists(sys.argv[1]):
        print("The file <{}> does not exist".format(sys.argv[1]))
        sys.exit()


def generate_output_name():
    path_to_file, filename, extension = split_filename(sys.argv[1])
    if path_to_file == "":
        new_name = filename + ".urdf"
    else:
        new_name = path_to_file + "/" + filename + ".urdf"
    return new_name


def check_output_file():
    output_name = generate_output_name()
    if os.path.exists(output_name):
        print("Warning: <{}> already exists. Do you want to continue and overwrite it? [y, n] > ".format(output_name), end="")
        ans = input().lower()
        if ans not in ["y", "yes"]:
            sys.exit()


def write_urdf_text():
    output_name = generate_output_name()
    _, name, _ = split_filename(sys.argv[1])
    print("Creation of <{}>...".format(output_name), end="")
    with open(output_name, "w") as f:
        text = """<?xml version="1.0" ?>
<robot name="{}.urdf">
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value="0.0"/>
       <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="white">
       <color rgba="1 1 1 1"/>
     </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
        <!-- You could also specify the collision (for the {}) with a "box" tag: -->
        <!-- <box size=".06 .06 .06"/> -->
      </geometry>
    </collision>
  </link>
</robot>
        """.format(name, name, name, name)
        f.write(text)
        print(" done")


if __name__ == "__main__":
    from commons import categories
    
    # shapenet_datadir = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/ShapeNetCore.v2/"
    shapenet_datadir = "/home/fudan248/zhangjinyu/code_repo/shapenetcorev2/shapenetcorev2_flipped"
    for synsetId in categories.keys():
        category_path = os.path.join(shapenet_datadir, synsetId)
        
        print("Creating urdf for category: ", synsetId)
        
        if not os.path.exists(category_path):
            print("Category path does not exist: ", category_path)
            continue
        
        for obj_id in os.listdir(category_path):
            try:
                obj_path = os.path.join(category_path, obj_id, "models", "model_normalized.obj")
                output_name = obj_path.replace(".obj", ".urdf")
                if not os.path.exists(os.path.dirname(output_name)):
                    os.makedirs(os.path.dirname(output_name))
                # print("Creating urdf for file: ", obj_path)
                name = "model_normalized"
                with open(output_name, "w") as f:
                    text = """<?xml version="1.0" ?>
    <robot name="{}.urdf">
    <link name="baseLink">
        <inertial>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <mass value="0.0"/>
        <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
        <visual>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
        </geometry>
        <material name="white">
        <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <mesh filename="{}.obj" scale="1.0 1.0 1.0"/>
            <!-- You could also specify the collision (for the {}) with a "box" tag: -->
            <!-- <box size=".06 .06 .06"/> -->
        </geometry>
        </collision>
    </link>
    </robot>
                    """.format(name, name, name, name)
                    f.write(text)
            except:
                print("Error in creating urdf for file: ", obj_path)
                
                