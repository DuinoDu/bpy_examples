import bpy
import math

# 清除默认场景中的所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 创建一个立方体
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "AnimatedCube"

# 设置动画总帧数
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 60
fps = 24
scene.render.fps = fps

# 设置第一个关键帧（起始位置）
cube.location = (0, 0, 0)
cube.rotation_euler = (0, 0, 0)
cube.keyframe_insert(data_path="location", frame=1)
cube.keyframe_insert(data_path="rotation_euler", frame=1)

# 设置第二个关键帧（20帧 - 立方体向上移动）
scene.frame_current = 20
cube.location = (0, 0, 5)
cube.keyframe_insert(data_path="location", frame=20)

# 设置第三个关键帧（40帧 - 立方体旋转并移动到另一个位置）
scene.frame_current = 40
cube.location = (5, 5, 0)
cube.rotation_euler = (0, 0, math.radians(180))
cube.keyframe_insert(data_path="location", frame=40)
cube.keyframe_insert(data_path="rotation_euler", frame=40)

# 设置第四个关键帧（60帧 - 立方体回到起始位置）
scene.frame_current = 60
cube.location = (0, 0, 0)
cube.rotation_euler = (0, 0, math.radians(360))
cube.keyframe_insert(data_path="location", frame=60)
cube.keyframe_insert(data_path="rotation_euler", frame=60)

# 改变插值方式为"贝塞尔曲线"使动画更平滑
for fcurve in cube.animation_data.action.fcurves:
    for kf in fcurve.keyframe_points:
        kf.interpolation = 'BEZIER'

# 回到第一帧
scene.frame_current = 1

print("简单的动画已创建完成！按播放按钮查看动画。")
