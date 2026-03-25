import os
import shutil

processed_data_dir = "assets/object/DGN_2k/processed_data"
raw_mesh_dir = "assets/object/DGN_obj_raw/DGN_obj/raw_mesh"
output_dir = "assets/object/DGN_2k/raw_mesh"

os.makedirs(output_dir, exist_ok=True)

# 1. processed_data 中的 object names（文件夹名）
processed_names = {
    name for name in os.listdir(processed_data_dir)
    if os.path.isdir(os.path.join(processed_data_dir, name))
}

# 2. raw_mesh 中的 object names（去掉 .obj 后缀）
raw_mesh_names = {
    os.path.splitext(name)[0]
    for name in os.listdir(raw_mesh_dir)
    if name.endswith(".obj")
}

# 3. 取交集
common_names = processed_names & raw_mesh_names

print(f"Processed objects : {len(processed_names)}")
print(f"Raw mesh objects  : {len(raw_mesh_names)}")
print(f"Common objects   : {len(common_names)}")

missing_in_raw = processed_names - raw_mesh_names
print("Missing in raw_mesh:", missing_in_raw)

# 4. 拷贝对应的 .obj 文件
for name in sorted(common_names):
    src = os.path.join(raw_mesh_dir, f"{name}.obj")
    dst = os.path.join(output_dir, f"{name}.obj")
    shutil.copy2(src, dst)

print(f"Copied {len(common_names)} .obj files to '{output_dir}'.")


