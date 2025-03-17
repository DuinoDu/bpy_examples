import bpy
import os
import math
from mathutils import Matrix, Vector, Quaternion, Euler
from pdebug.utils.fileio import load_content
from pdebug.utils.bpy import load_gltf
import torch
import numpy as np

# bpy.context.preferences.view.show_splash = False
# bpy.ops.wm.save_userpref()


def compute_rotation_from_vectors_bpy(vec1, vec2):
    """
    Compute rotation matrix from two vectors.
    """
    vec1 = Vector(vec1).normalized()
    vec2 = Vector(vec2).normalized()

    dot_product = vec1.dot(vec2)
    if abs(dot_product) > 0.9999:
        if dot_product > 0:
            # vec1 & vec2 are in same direction
            return Matrix.Identity(4)
        else:
            # vec1 & vec2 are in 180-direction
            ref_vec = Vector((0, 1, 0))
            if abs(vec2.dot(ref_vec)) > 0.9:
                ref_vec = Vector((1, 0, 0))
            axis = vec2.cross(ref_vec).normalized()
            if axis.length < 1e-6:
                axis = Vector((0, 0, 1))
            return Quaternion(axis, math.pi).to_matrix().to_4x4()

    axis = vec1.cross(vec2).normalized()   
    angle = math.acos(dot_product)
    rotation_matrix = Matrix.Rotation(angle, 4, axis)
    return rotation_matrix


def bpy_vector_to_pytorch_tensor(vec):
    return torch.tensor([vec.x, vec.y, vec.z], dtype=torch.float32).unsqueeze(0)

def pytorch_tensor_to_bpy_matrix(tensor):
    return Matrix(T_torch.squeeze().numpy().tolist())


class RotationFromVectors(torch.nn.Module):

    def __init__(self):
        super(RotationFromVectors, self).__init__()

    def forward(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix from two vectors using PyTorch.
        
        Args:
            vec1: tensor of shape (..., 3)
            vec2: tensor of shape (..., 3)
        Returns:
            rotation_matrix: tensor of shape (..., 4, 4)
        """
        # Normalize vectors
        vec1 = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        vec2 = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
        
        # Compute dot product
        dot_product = torch.sum(vec1 * vec2, dim=-1)
        
        # Create identity matrix for output
        batch_shape = vec1.shape[:-1]
        identity = torch.eye(4, device=vec1.device).expand(*batch_shape, 4, 4)
        
        # Handle parallel vectors (dot product close to 1 or -1)
        parallel_mask = torch.abs(dot_product) > 0.9999
        same_dir_mask = dot_product > 0
        
        # Handle anti-parallel case
        anti_parallel_mask = parallel_mask & ~same_dir_mask
        if torch.any(anti_parallel_mask):
            # Create reference vector
            ref_vec = torch.tensor([0., 1., 0.], device=vec1.device)
            ref_dot = torch.sum(vec2 * ref_vec, dim=-1)
            alt_ref_mask = torch.abs(ref_dot) > 0.9
            ref_vec = torch.where(alt_ref_mask.unsqueeze(-1),
                                torch.tensor([1., 0., 0.], device=vec1.device),
                                ref_vec)
            
            # Compute rotation axis
            axis = torch.cross(vec2, ref_vec.expand_as(vec2))
            axis_norm = torch.norm(axis, dim=-1, keepdim=True)
            axis = torch.where(axis_norm < 1e-6,
                             torch.tensor([0., 0., 1.], device=vec1.device).expand_as(axis),
                             axis / axis_norm)
            
            # Create rotation matrix for pi radians
            angle = torch.full_like(dot_product, torch.pi)
            c = torch.cos(angle)
            s = torch.sin(angle)
            t = 1.0 - c
            
            x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
            
            rot_matrix = torch.stack([
                t*x*x + c,    t*x*y - z*s,  t*x*z + y*s,  torch.zeros_like(x),
                t*x*y + z*s,  t*y*y + c,    t*y*z - x*s,  torch.zeros_like(x),
                t*x*z - y*s,  t*y*z + x*s,  t*z*z + c,    torch.zeros_like(x),
                torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)
            ], dim=-1).reshape(*batch_shape, 4, 4)
            
            identity = torch.where(anti_parallel_mask.unsqueeze(-1).unsqueeze(-1),
                                 rot_matrix, identity)
        
        # Handle non-parallel case
        non_parallel_mask = ~parallel_mask
        if torch.any(non_parallel_mask):
            axis = torch.cross(vec1, vec2)
            axis = axis / torch.norm(axis, dim=-1, keepdim=True)
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            c = torch.cos(angle)
            s = torch.sin(angle)
            t = 1.0 - c
            
            x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
            
            rot_matrix = torch.stack([
                t*x*x + c,    t*x*y - z*s,  t*x*z + y*s,  torch.zeros_like(x),
                t*x*y + z*s,  t*y*y + c,    t*y*z - x*s,  torch.zeros_like(x),
                t*x*z - y*s,  t*y*z + x*s,  t*z*z + c,    torch.zeros_like(x),
                torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)
            ], dim=-1).reshape(*batch_shape, 4, 4)
            
            identity = torch.where(non_parallel_mask.unsqueeze(-1).unsqueeze(-1),
                                 rot_matrix, identity)
        
        return identity


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

kps_data = load_content("./animation_transform.json")
skeleton_relationship = load_content("./skeleton_relationship.json")
relations = dict()
for relation in skeleton_relationship:
    parent, child = relation
    relations[child] = parent

root = os.path.expanduser("~/code/bpy_examples")
gltf_path = f"{root}/test_data/robot_v1.gltf"
load_gltf(filepath=gltf_path, location=(0, 3, 0), name="robot_v1")
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
armature_obj = armatures[0]

# clean existing animation
if armature_obj.animation_data:
    armature_obj.animation_data.action = None
    armature_obj.animation_data_clear()
# create new animation
armature_obj.animation_data_create()
action = bpy.data.actions.new("ComputedAction")
action.use_frame_range = True
action.frame_start = 1
action.frame_end = len(kps_data["frames"])
armature_obj.animation_data.action = action
# get rest matrix
rest_matrix = {}
for bone in armature_obj.data.bones:
    rest_matrix[bone.name] = bone.matrix.copy()
    # patch for rest_matrix
    if "left thigh" in bone.name:
        rest_matrix[bone.name] = Euler((-3.14, 0, 0), "XYZ").to_matrix().to_4x4()

joints = {}
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(kps_data["frames"])

joints_armature = bpy.data.armatures.new(name="JointsArmature")
joints_armature_obj = bpy.data.objects.new(name="JointsArmatureObj", object_data=joints_armature)
bpy.context.collection.objects.link(joints_armature_obj)
joints_armature_obj.location = (0, -2, 0)    # move to left
joints_armature_obj.animation_data_create()
action = bpy.data.actions.new("ComputedAction2")
action.use_frame_range = True
action.frame_start = 1
action.frame_end = len(kps_data["frames"])
armature_obj.animation_data.action = action

rotate_module = RotationFromVectors()
torch.save(rotate_module, "tmp_rotate_op.pt")
rotate_op = torch.load("tmp_rotate_op.pt", weights_only=False)
rotate_op.eval()


for frame_idx, frame in enumerate(kps_data["frames"]):
    current_frame = frame_idx + 1
    bpy.context.scene.frame_set(current_frame)

    # for first frame, create bone in joints_armature_obj
    if frame_idx == 0:
        for name in frame["joints"]:
            bpy.context.view_layer.objects.active = joints_armature_obj
            bpy.ops.object.mode_set(mode='EDIT')
            new_bone = joints_armature.edit_bones.new(name=name)
            new_bone.head = Vector(frame["joints"][name][0])
            new_bone.tail = Vector(frame["joints"][name][1])
            bpy.ops.object.mode_set(mode='OBJECT')
        for name in frame["joints"]:
            parent_name = relations.get(name, None)
            if parent_name is not None:
                bpy.ops.object.mode_set(mode='EDIT')
                bone = joints_armature.edit_bones.get(name)
                bone.parent = joints_armature.edit_bones.get(parent_name)
                bpy.ops.object.mode_set(mode='OBJECT')

    for name in frame["joints"]:
        # add keyframe to joints
        for i, suffix in enumerate(['head', 'tail']):
            joint_name = f"{name}_{suffix}"
            if joint_name not in joints:
                bpy.ops.object.empty_add(type='SPHERE', radius=0.1, location=(0,0,0))
                joints[joint_name] = bpy.context.active_object
                joints[joint_name].name = joint_name
                if not joints[joint_name].animation_data:
                    joints[joint_name].animation_data_create()
            joints[joint_name].location = frame["joints"][name][i]
            joints[joint_name].keyframe_insert(data_path="location", frame=current_frame)
        
        # add keyframe to armature_obj
        child_name = name
        parent_name = relations.get(child_name, None)

        if parent_name is None:
            matrix_basis = Matrix.Identity(4)
        else:
            child_head, child_tail = frame["joints"][child_name]
            child_vec = (Vector(child_tail) - Vector(child_head)).normalized()
            parent_head, parent_tail = frame["joints"][parent_name]
            parent_vec = (Vector(parent_tail) - Vector(parent_head)).normalized()

            T_parent_child = compute_rotation_from_vectors_bpy(child_vec, parent_vec)

            # BEGIN: pytorch test
            T_torch = rotate_op(bpy_vector_to_pytorch_tensor(child_vec),
                                bpy_vector_to_pytorch_tensor(parent_vec))
            T_torch = pytorch_tensor_to_bpy_matrix(T_torch)
            np.testing.assert_allclose(np.array(T_parent_child), np.array(T_torch), atol=1e-6)
            # END: pytorch test

            matrix_basis = rest_matrix[name].to_4x4().inverted() @ T_parent_child

        loc, rot_quat, scale = matrix_basis.decompose()
        for bone in [joints_armature_obj.pose.bones.get(name), armature_obj.pose.bones.get(name)]:
            bone.location = loc
            bone.rotation_quaternion = rot_quat
            bone.scale = scale
            bone.keyframe_insert("location", frame=current_frame)
            bone.keyframe_insert("rotation_quaternion", frame=current_frame)
            bone.keyframe_insert("scale", frame=current_frame)

bpy.context.scene.frame_set(1)
bpy.ops.object.select_all(action="DESELECT")
