import bpy
import math
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# OpenPose keypoint definitions
# Based on BODY_25 keypoint format:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
KEYPOINTS = {
    'Nose': (0, 0, 0),
    'Neck': (0, 0, -0.2),
    'RShoulder': (0.1, 0, -0.2),
    'RElbow': (0.2, 0, -0.3),
    'RWrist': (0.3, 0, -0.4),
    'LShoulder': (-0.1, 0, -0.2),
    'LElbow': (-0.2, 0, -0.3),
    'LWrist': (-0.3, 0, -0.4),
    'MidHip': (0, 0, -0.6),
    'RHip': (0.1, 0, -0.6),
    'RKnee': (0.1, 0, -1.0),
    'RAnkle': (0.1, 0, -1.4),
    'LHip': (-0.1, 0, -0.6),
    'LKnee': (-0.1, 0, -1.0),
    'LAnkle': (-0.1, 0, -1.4),
    'REye': (0.03, 0, 0.03),
    'LEye': (-0.03, 0, 0.03),
    'REar': (0.06, 0, 0),
    'LEar': (-0.06, 0, 0),
    'LBigToe': (-0.12, 0, -1.45),
    'LSmallToe': (-0.14, 0, -1.45),
    'LHeel': (-0.08, 0, -1.42),
    'RBigToe': (0.12, 0, -1.45),
    'RSmallToe': (0.14, 0, -1.45),
    'RHeel': (0.08, 0, -1.42),
}

# Define connections between keypoints to form the skeleton
CONNECTIONS = [
    ('Nose', 'Neck'),
    ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('Neck', 'MidHip'),
    ('MidHip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('MidHip', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('Nose', 'REye'), ('REye', 'REar'),
    ('Nose', 'LEye'), ('LEye', 'LEar'),
    ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'), ('RBigToe', 'RSmallToe'),
    ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'), ('LBigToe', 'LSmallToe')
]

# Create an armature
armature = bpy.data.armatures.new("Skeleton_Armature")
obj = bpy.data.objects.new("Skeleton", armature)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')

# Create the bones
edit_bones = armature.edit_bones

bone_refs = {}
for parent_name, child_name in CONNECTIONS:
    bone_name = f"{parent_name}_{child_name}"
    if bone_name not in bone_refs:
        bone = edit_bones.new(bone_name)
        bone.head = Vector(KEYPOINTS[parent_name])
        bone.tail = Vector(KEYPOINTS[child_name])
        bone_refs[bone_name] = bone

# Go to pose mode to set up animation
bpy.ops.object.mode_set(mode='POSE')

# Define walking animation keyframes
frames_per_step = 20
stride_length = 0.3
step_height = 0.1
total_frames = frames_per_step * 2  # Complete walk cycle

# Foot locations for keyframes
def set_foot_keyframe(frame, bone_name, locZ, influence=1.0):
    pose_bone = obj.pose.bones.get(bone_name)
    if pose_bone:
        pose_bone.location.z = locZ * influence
        pose_bone.keyframe_insert(data_path="location", frame=frame)


# Leg angles for keyframes
def set_leg_angle(frame, knee_name, angle):
    pose_bone = obj.pose.bones.get(knee_name)
    if pose_bone:
        pose_bone.rotation_euler.y = angle
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)


# Animation Setup
for frame in range(total_frames + 1):
    bpy.context.scene.frame_set(frame)
    
    # Animation stage (0 to 1)
    t = frame / total_frames
    
    # Alternating leg movement
    left_phase = math.sin(t * 2 * math.pi)
    right_phase = math.sin((t * 2 * math.pi) + math.pi)  # 180° out of phase
    
    # Set left leg keyframes
    left_foot_height = step_height * max(0, left_phase)
    set_foot_keyframe(frame, "LKnee_LAnkle", left_foot_height)
    set_foot_keyframe(frame, "LAnkle_LBigToe", left_foot_height * 1.2)
    set_foot_keyframe(frame, "LBigToe_LSmallToe", left_foot_height * 1.2)
    set_foot_keyframe(frame, "LAnkle_LHeel", left_foot_height * 0.8)
    
    # Set right leg keyframes
    right_foot_height = step_height * max(0, right_phase)
    set_foot_keyframe(frame, "RAnkle", right_foot_height)
    set_foot_keyframe(frame, "RBigToe", right_foot_height * 1.2)
    set_foot_keyframe(frame, "RSmallToe", right_foot_height * 1.2)
    set_foot_keyframe(frame, "RHeel", right_foot_height * 0.8)
    
    # Knee bend
    set_leg_angle(frame, "LKnee", max(0, left_phase) * 0.5)
    set_leg_angle(frame, "RKnee", max(0, right_phase) * 0.5)
    
    # Arms swing in opposite phase to legs
    arm_swing = 0.2
    set_leg_angle(frame, "LElbow", -max(0, right_phase) * 0.3)  # Opposite to right leg
    set_leg_angle(frame, "RElbow", -max(0, left_phase) * 0.3)   # Opposite to left leg

# Set up looping animation
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = total_frames
bpy.context.scene.render.fps = 24

# Return to object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Select the armature
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

print("Skeleton character with OpenPose-style bones created successfully!")
