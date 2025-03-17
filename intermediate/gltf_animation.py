"""
It is not easy to manipulate gltf animation data directly.
"""
import struct
import base64
from collections import defaultdict
from pygltflib import GLTF2, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget, Buffer, BufferView, Accessor, BufferFormat
from pygltflib import FLOAT, VEC3, VEC4, SCALAR


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


def main():
    root = "/Users/bytedance/code/bpy_examples"
    gltf_path = f"{root}/test_data/robot_v1.gltf"
    gltf_animation = GltfAnimation(gltf_path)
    joints_matrix = defaultdict(list) 
    
    # joints_matrix[name].append(T_parent_child)
    
    for name, matrix_list in joints_matrix.items():
        node_idx = gltf_animation.name2id(name)
        gltf_animation.add_animation(node_idx, matrix_list)
    new_gltf = f"{root}/../../Downloads/robot_v1_updated.gltf"
    gltf_animation.save(new_gltf)
    gltf_animation.print_animation()


if __name__ == "__main__":
    main()
