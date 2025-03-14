import bpy
import math
from pdebug.utils.fileio import load_content
from pdebug.utils.bpy import load_gltf
from mathutils import Matrix, Vector, Quaternion
import struct
import base64
import numpy as np
from collections import defaultdict
from pygltflib import GLTF2, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget, Buffer, BufferView, Accessor, BufferFormat
from pygltflib import FLOAT, VEC3, VEC4, SCALAR

# bpy.context.preferences.view.show_splash = False
# bpy.ops.wm.save_userpref()


def computeRotationFromVectors(vec1, vec2):
    vec1 = Vector(vec1).normalized()
    vec2 = Vector(vec2).normalized()

    dot_product = vec1.dot(vec2)
    if abs(dot_product) > 0.9999:
        if dot_product > 0:
            # vec1 & vec2 are in same direction
            return Matrix.Identity(4)
        else:
            # vec1 & vec2 are in 180-direction
            if abs(vec1.x) > 1e-6:
                axis = Vector((1, 0, 0)).cross(vec1).normalized()
            elif abs(vec1.y) > 1e-6:
                axis = Vector((0, 1, 0)).cross(vec1).normalized()
            else:
                axis = Vector((0, 0, 1)).cross(vec1).normalized()
            return Quaternion(axis, math.pi).to_matrix().to_4x4()
            # perp_axis = vec1.orthogonal().normalized()
            # return Matrix.Rotation(math.pi, 4, perp_axis)

    axis = vec1.cross(vec2).normalized()   
    angle = math.acos(dot_product)
    rotation_matrix = Matrix.Rotation(angle, 4, axis)
    return rotation_matrix


class GltfAnimation:
    def __init__(self, gltf_path):
        self.gltf = GLTF2().load(gltf_path)
        self._node_name_to_id = { node.name: idx for idx, node in enumerate(self.gltf.nodes)}

        self.buffer_data = bytearray()
        self.time_accessor_list = []
        self.transform_accessor_list = []

        self.print_animation()
        self.clear_animation()
    
    def clear_animation(self):
        self.gltf.animations.clear()
        self.gltf.accessors = self.gltf.accessors[:7]
        # self.gltf.buffers.clear()
        # self.gltf.bufferViews.clear()
        # self.gltf.accessors.clear()

        # self.buffer_idx = 0
        # self.main_buffer = Buffer(byteLength=0)
        # self.gltf.buffers.append(self.main_buffer)
    
    def name2id(self, name):
        return self._node_name_to_id[name]

    def names(self):
        return list(self._node_name_to_id.keys())
    
    def add_animation(self, node_idx, matrix_list):
        time_data = bytearray()
        trans_data = bytearray()
        rot_data = bytearray()
        scale_data = bytearray()
        
        times = [i for i in range(len(matrix_list))]
        for t, matrix in enumerate(matrix_list):
            t = t * 1. / 24
            time_data.extend(struct.pack("f", t))

            trans, rot_wxyz, scale = matrix.decompose()
            rot_xyzw = [rot_wxyz.x, rot_wxyz.y, rot_wxyz.z, rot_wxyz.w]
            trans_data.extend(struct.pack("fff", *trans))
            rot_data.extend(struct.pack("ffff", *rot_xyzw))
            scale_data.extend(struct.pack("fff", *scale))
        
        time_offset = len(self.buffer_data)
        self.buffer_data.extend(time_data)
        trans_offset = len(self.buffer_data)
        self.buffer_data.extend(trans_data)
        rot_offset = len(self.buffer_data)
        self.buffer_data.extend(rot_data)
        scale_offset = len(self.buffer_data)
        self.buffer_data.extend(scale_data)
        
        # extend bufferViews
        buffer_idx = len(self.gltf.buffers)
        time_view_idx = len(self.gltf.bufferViews)
        trans_view_idx = len(self.gltf.bufferViews) + 1
        rot_view_idx = len(self.gltf.bufferViews) + 2
        scale_view_idx = len(self.gltf.bufferViews) + 3

        self.gltf.bufferViews.extend([
            BufferView(buffer=buffer_idx, byteOffset=time_offset, byteLength=len(time_data)),
            BufferView(buffer=buffer_idx, byteOffset=trans_offset, byteLength=len(trans_data)),
            BufferView(buffer=buffer_idx, byteOffset=rot_offset, byteLength=len(rot_data)),
            BufferView(buffer=buffer_idx, byteOffset=scale_offset, byteLength=len(scale_data))
            ])

        # extend accessors
        time_acc_idx = len(self.gltf.accessors)
        trans_acc_idx = len(self.gltf.accessors) + 1
        rot_acc_idx = len(self.gltf.accessors) + 2
        scale_acc_idx = len(self.gltf.accessors) + 3

        self.gltf.accessors.extend([
            Accessor(bufferView=time_view_idx, componentType=FLOAT, count=len(times), type=SCALAR, min=[min(times)], max=[max(times)]),
            Accessor(bufferView=trans_view_idx, componentType=FLOAT, count=len(times), type=VEC3),
            Accessor(bufferView=rot_view_idx, componentType=FLOAT, count=len(times), type=VEC4),
            Accessor(bufferView=scale_view_idx, componentType=FLOAT, count=len(times), type=VEC3)
        ])
        
        # create animation
        animation = Animation()
        animation.samplers.extend([
            AnimationSampler(input=time_acc_idx, output=trans_acc_idx, interpolation="LINEAR"),
            AnimationSampler(input=time_acc_idx, output=rot_acc_idx, interpolation="LINEAR"),
            AnimationSampler(input=time_acc_idx, output=scale_acc_idx, interpolation="LINEAR")
        ])
        animation.channels.extend([
            AnimationChannel(sampler=0, target=AnimationChannelTarget(node=node_idx, path="translation")),
            AnimationChannel(sampler=1, target=AnimationChannelTarget(node=node_idx, path="rotation")),
            AnimationChannel(sampler=2, target=AnimationChannelTarget(node=node_idx, path="scale"))
        ])
        self.gltf.animations.append(animation)

    def print_animation(self):
        for animation_idx, animation in enumerate(self.gltf.animations):
            for channel_idx, channel in enumerate(animation.channels):
                sampler = animation.samplers[channel.sampler]

                input_accessor_idx = sampler.input
                input_accessor = self.gltf.accessors[input_accessor_idx]

                output_accessor_idx = sampler.output
                output_accessor = self.gltf.accessors[output_accessor_idx]

                input_view = self.gltf.bufferViews[input_accessor.bufferView]
                output_view = self.gltf.bufferViews[output_accessor.bufferView]

                input_buffer = self.gltf.buffers[input_view.buffer]
                output_buffer = self.gltf.buffers[input_view.buffer]

                # input_data = GLTF2.decode_buffer(
                #     input_buffer.data,
                #     input_accessor.count,
                #     input_accessor.componentType,
                #     input_accessor.type,
                #     buffer_format=BufferFormat.NUMPY
                # )

                # output_data = GLTF2.decode_buffer(
                #     output_buffer.data,
                #     output_accessor.count,
                #     output_accessor.componentType,
                #     output_accessor.type,
                #     buffer_format=BufferFormat.NUMPY
                # )

                target_node_idx = channel.target.node
                target_path = channel.target.path
                print(animation_idx, channel_idx, self.gltf.nodes[target_node_idx].name, ", ", target_path)

    def save(self, output):
        buffer = Buffer(byteLength=len(self.buffer_data))
        buffer.uri = f"data:application/octet-stream;base64,{base64.b64encode(self.buffer_data).decode('ascii')}"
        self.gltf.buffers.append(buffer)
        self.gltf.save(output)


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

data = load_content("./animation_transform.json")
skeleton_relationship = load_content("./skeleton_relationship.json")
relations = dict()
for relation in skeleton_relationship:
    parent, child = relation
    relations[child] = parent

root = "/Users/bytedance/code/bpy_examples"
gltf_path = f"{root}/test_data/robot_v1.gltf"
# gltf_animation = GltfAnimation(gltf_path)
# joints_matrix = defaultdict(list) 
load_gltf(filepath=gltf_path, location=(0, 3, 0), name="robot_v1")
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
armature_obj = armatures[0]

if armature_obj.animation_data:
    armature_obj.animation_data.action = None  # 清除绑定的动作
    armature_obj.animation_data_clear()       # 清除动画数据
# create new animation
armature_obj.animation_data_create()
action = bpy.data.actions.new("ComputedAction")
action.use_frame_range = True
action.frame_start = 1
action.frame_end = len(data["frames"])
armature_obj.animation_data.action = action
# get rest matrix
rest_matrix = {}
for bone in armature_obj.data.bones:
    rest_matrix[bone.name] = bone.matrix.copy()

joints = {}
frame_start = 1
frame_end = len(data["frames"])
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

joints_armature = bpy.data.armatures.new(name="JointsArmature")
joints_armature_obj = bpy.data.objects.new(name="JointsArmatureObj", object_data=joints_armature)
bpy.context.collection.objects.link(joints_armature_obj)
joints_armature_obj.location = (0, -2, 0)    # move to left

for frame_idx, frame in enumerate(data["frames"]):
    bpy.context.scene.frame_set(frame_idx + 1)
    for name in frame["joints"]:
        for i, suffix in enumerate(['head', 'tail']):
            joint_name = f"{name}_{suffix}"
            if joint_name not in joints:
                bpy.ops.object.empty_add(type='SPHERE', radius=0.1, location=(0,0,0))
                joints[joint_name] = bpy.context.active_object
                joints[joint_name].name = joint_name
            joints[joint_name].location = frame["joints"][name][i]
            joints[joint_name].keyframe_insert(data_path="location")
            
        if frame_idx == 0:
            bpy.context.view_layer.objects.active = joints_armature_obj
            bpy.ops.object.mode_set(mode='EDIT')
            new_bone = joints_armature.edit_bones.new(name=name)
            new_bone.head = Vector(frame["joints"][name][0])
            new_bone.tail = Vector(frame["joints"][name][1])
            bpy.ops.object.mode_set(mode='OBJECT')

        child_name = name
        parent_name = relations.get(child_name, None)

        if parent_name is None:
            matrix_basis = Matrix.Identity(4)
        else:
            child_head, child_tail = frame["joints"][child_name]
            child_vec = (Vector(child_tail) - Vector(child_head)).normalized()
            parent_head, parent_tail = frame["joints"][parent_name]
            parent_vec = (Vector(parent_tail) - Vector(parent_head)).normalized()
            T_parent_child = computeRotationFromVectors(child_vec, parent_vec)
            matrix_basis = rest_matrix[name].to_4x4().inverted() @ T_parent_child

        if "thigh" in name:
            # TODO(min.du): fix bug
            matrix_basis = Matrix.Identity(4)

        bone = armature_obj.pose.bones.get(name)
        loc, rot_quat, scale = matrix_basis.decompose()
        bone.location = loc
        bone.rotation_quaternion = rot_quat
        bone.scale = scale
        bone.keyframe_insert("location")
        bone.keyframe_insert("rotation_quaternion")
        bone.keyframe_insert("scale")

        # joints_matrix[name].append(T_parent_child)
    

# for name, matrix_list in joints_matrix.items():
#     node_idx = gltf_animation.name2id(name)
#     gltf_animation.add_animation(node_idx, matrix_list)
# new_gltf = f"{root}/../../Downloads/robot_v1_updated.gltf"
# gltf_animation.save(new_gltf)
# gltf_animation.print_animation()
# load_gltf(filepath=new_gltf, location=(0, 3, 0))

bpy.context.scene.frame_set(frame_start)
bpy.ops.object.select_all(action="DESELECT")
