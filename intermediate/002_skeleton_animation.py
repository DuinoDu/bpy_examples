import bpy
import os
import numpy as np
import json
from mathutils import Matrix, Vector 

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


def load_gltf_into_scene(gltf_path):
    # 导入GLTF文件
    if gltf_path.lower().endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=gltf_path)
    elif gltf_path.lower().endswith('.glb'):
        bpy.ops.import_scene.glb(filepath=gltf_path)
    else:
        print("错误：文件必须是.gltf或.glb格式")
        return
    
    print(f"已成功导入: {os.path.basename(gltf_path)}")


def print_bone_hierarchy(armature_obj=None):
    if armature_obj is None:
        armature_obj = bpy.context.active_object

    if armature_obj.type != 'ARMATURE':
        print(f"错误: 选中的对象 '{armature_obj.name}' 不是骨架")
        return
    
    armature = armature_obj.data
    print(f"\n骨架 '{armature_obj.name}' 的层级结构:")
    
    # 获取根骨骼(没有父骨骼的骨骼)
    root_bones = [bone for bone in armature.bones if bone.parent is None]
    
    # 递归打印骨骼层级
    for root_bone in root_bones:
        print_bone_tree(root_bone)
        

def print_bone_tree(bone, indent=0):
    """
    递归打印骨骼及其子骨骼
    
    参数:
        bone: 骨骼对象
        indent: 缩进级别
    """
    # 打印当前骨骼，带缩进
    print("  " * indent + f"|- {bone.name}")
    
    # 递归打印所有子骨骼
    for child in bone.children:
        print_bone_tree(child, indent + 1)


def get_bone_hierarchy_dict(armature_obj=None):
    """
    获取骨架中骨骼的层级结构，并以字典形式返回
    
    参数:
        armature_obj: 骨架对象，如果为None则使用当前活动对象
        
    返回:
        骨骼层级的嵌套字典
    """
    # 如果没有指定骨架，则使用当前活动对象
    if armature_obj is None:
        armature_obj = bpy.context.active_object
        
    # 确保是骨架对象
    if armature_obj.type != 'ARMATURE':
        print(f"错误: 选中的对象 '{armature_obj.name}' 不是骨架")
        return {}
    
    armature = armature_obj.data
    
    # 获取根骨骼(没有父骨骼的骨骼)
    root_bones = [bone for bone in armature.bones if bone.parent is None]
    
    # 构建层级字典
    hierarchy = {}
    for root_bone in root_bones:
        hierarchy[root_bone.name] = build_bone_dict(root_bone)
        
    return hierarchy


def build_bone_dict(bone):
    """
    递归构建骨骼字典
    
    参数:
        bone: 骨骼对象
        
    返回:
        包含子骨骼的字典
    """
    result = {}
    for child in bone.children:
        result[child.name] = build_bone_dict(child)
    return result


def get_bone_parent_child_list(armature_obj=None):
    """
    获取骨骼的父子关系列表
    
    参数:
        armature_obj: 骨架对象，如果为None则使用当前活动对象
        
    返回:
        包含(parent_name, child_name)元组的列表
    """
    # 如果没有指定骨架，则使用当前活动对象
    if armature_obj is None:
        armature_obj = bpy.context.active_object
        
    # 确保是骨架对象
    if armature_obj.type != 'ARMATURE':
        print(f"错误: 选中的对象 '{armature_obj.name}' 不是骨架")
        return []
    
    armature = armature_obj.data
    relationships = []
    
    # 遍历所有骨骼
    for bone in armature.bones:
        if bone.parent:
            relationships.append((bone.parent.name, bone.name))
    
    return relationships

    # # 打印当前选中骨架的层级结构
    # print_bone_hierarchy()
    # 
    # # 获取骨骼层级字典
    # hierarchy_dict = get_bone_hierarchy_dict()
    # print("\n骨骼层级字典:")
    # print(hierarchy_dict)
    # 
    # # 获取父子关系列表
    # parent_child_list = get_bone_parent_child_list()
    # print("\n骨骼父子关系列表:")
    # for parent, child in parent_child_list:
    #     print(f"父骨骼: {parent} -> 子骨骼: {child}")


def get_armature_animation_data(armature_obj=None):
    """
    获取骨架动画数据，包括每个骨骼的矩阵和动画信息
    
    参数:
        armature_obj: 骨架对象，如果为None则使用当前活动对象
        
    返回:
        包含骨骼动画数据的字典
    """
    # 如果没有指定骨架，则使用当前活动对象
    if armature_obj is None:
        armature_obj = bpy.context.active_object
        
    # 确保是骨架对象
    if armature_obj.type != 'ARMATURE':
        print(f"错误: 选中的对象 '{armature_obj.name}' 不是骨架")
        return {}
    
    # 获取骨架数据
    armature = armature_obj.data
    
    # 创建结果字典
    result = {
        "armature_name": armature_obj.name,
        "bones": {},
        "animations": []
    }
    
    # 收集骨骼信息
    for bone in armature.bones:
        # 获取骨骼的各种矩阵
        if bone.parent:
            local_matrix = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            local_matrix = bone.matrix_local
            
        # 转换矩阵为列表格式
        matrix_world = armature_obj.matrix_world @ bone.matrix_local
        
        # 添加到结果字典
        result["bones"][bone.name] = {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent else None,
            "head": [round(v, 6) for v in bone.head_local],
            "tail": [round(v, 6) for v in bone.tail_local],
            "matrix_local": matrix_to_list(bone.matrix_local),
            "matrix_world": matrix_to_list(matrix_world),
            "local_matrix": matrix_to_list(local_matrix),
        }
    
    # 检查是否有动画数据
    if armature_obj.animation_data and armature_obj.animation_data.action:
        action = armature_obj.animation_data.action
        
        # 确定帧范围
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        
        # 遍历所有帧
        for frame in range(frame_start, frame_end + 1):
            # 设置当前帧
            bpy.context.scene.frame_set(frame)
            
            frame_data = {
                "frame": frame,
                "time": frame / bpy.context.scene.render.fps,
                "bone_poses": {}
            }
            
            # 获取每个骨骼在当前帧的姿态
            for bone in armature_obj.pose.bones:
                pose_matrix = bone.matrix
                if bone.parent:
                    pose_local = bone.parent.matrix.inverted() @ pose_matrix
                else:
                    pose_local = pose_matrix
                
                frame_data["bone_poses"][bone.name] = {
                    "matrix": matrix_to_list(pose_matrix),
                    "matrix_local": matrix_to_list(pose_local),
                    "location": [round(v, 6) for v in bone.location],
                    "rotation_quaternion": [round(v, 6) for v in bone.rotation_quaternion],
                    "scale": [round(v, 6) for v in bone.scale]
                }
                bone.location = Vector((0,0,0))
            
            result["animations"].append(frame_data)
    
    return result


def matrix_to_list(matrix):
    return [[round(matrix[row][col], 6) for col in range(4)] for row in range(4)]


def dump_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"已导出到: {file_path}")


def export_armature_transforms(armature_name, output_path, frame_start=None, frame_end=None):
    armature = bpy.data.objects.get(armature_name)
    if not armature or armature.type != 'ARMATURE':
        raise ValueError(f"Armature '{armature_name}' not found or is not an armature.")

    # 初始化 JSON 数据结构
    data = {
        "armature_name": armature_name,
        "frames": []
    }
    
    if frame_start is None:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        frame_data = {"frame": frame, "bones": {}, "joints": {}}

        for bone in armature.pose.bones:
            matrix = bone.matrix_basis
            matrix_data = [list(row) for row in matrix]
            frame_data["bones"][bone.name] = matrix_data
            frame_data["joints"][bone.name] = [bone.head.to_tuple(), bone.tail.to_tuple()]

        data["frames"].append(frame_data)

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Armature transform data exported to {output_path}")


def apply_armature_transforms_from_json(armature_name, json_path):
    # get armature
    armature = bpy.data.objects.get(armature_name)
    if not armature or armature.type != 'ARMATURE':
        raise ValueError(f"Armature '{armature_name}' not found or is not an armature.")
    
    # load json
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, "r") as file:
        data = json.load(file)

    # remove existing animation
    if armature.animation_data:
        armature.animation_data.action = None  # 清除绑定的动作
        armature.animation_data_clear()       # 清除动画数据
    
    # create new animation
    armature.animation_data_create()
    action = bpy.data.actions.new("ImportedMotion")
    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = len(data["frames"])
    armature.animation_data.action = action

    # set animation data to bone
    for frame_data in data["frames"]:
        frame = frame_data["frame"]
        bpy.context.scene.frame_set(frame)  # 设置当前帧

        for bone_name, matrix_data in frame_data["bones"].items():
            bone = armature.pose.bones.get(bone_name)
            if not bone:
                print(f"Bone '{bone_name}' not found in armature. Skipping...")
                continue
            
            loc, rot_quat, scale = Matrix(matrix_data).decompose()
            bone.location = loc
            bone.rotation_quaternion = rot_quat
            bone.scale = scale
            bone.keyframe_insert("location")
            bone.keyframe_insert("rotation_quaternion")
            bone.keyframe_insert("scale")

    bpy.context.scene.frame_set(1)
    print(f"Armature transform data applied from {json_path}")


def main():
    """Show gltf and skeleton in blender."""
    # gltf_path = os.path.join(os.path.dirname(__file__), "../test_data/stickman/scene.gltf")
    root = "/Users/bytedance/code/bpy_examples"
    # gltf_path = f"{root}/test_data/stickman/scene.gltf"
    gltf_path = f"{root}/../../Downloads/robot_v1.gltf"

    load_gltf_into_scene(gltf_path)
    
    # 查找骨架(Armature)对象
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    
    if not armatures:
        print("模型中未找到骨架")
        return
    
    # 取消选择所有对象
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.object.mode_set(mode='POSE')
    
    for armature in armatures:
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # print_bone_hierarchy()
 
        # 获取父子关系列表
        parent_child_list = get_bone_parent_child_list()
        print("\nskeleton relation:")
        for parent, child in parent_child_list:
            print(f"  {parent} -> {child}")
        dump_to_json(parent_child_list, "skeleton_relationship.json")

        # data = get_armature_animation_data(armature)
        # dump_to_json(data, "animation_data.json")
        
        export_armature_transforms(armature.name, "animation_transform.json")
        apply_armature_transforms_from_json(armature.name, "animation_transform.json")

    print("处理完成")


if __name__ == "__main__":
    main()
