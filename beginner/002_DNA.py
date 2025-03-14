import bpy
import math

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

for i in range(2):
    bpy.ops.curve.primitive_bezier_curve_add(radius=1)
    curve =  bpy.context.active_object  
    curve.name = f"Helix_{i+1}"

    points = curve.data.splines[0].bezier_points
    for j in range(len(points)):
        angle = j * math.pi / 4
        offset = 0.2 if i == 0 else -0.2
        x = math.cos(angle) * (1 + offset)
        y = math.sin(angle) * (1 + offset)
        z = j * 0.5
        points[j].co = (x, y, z)

    curve.animation_data_create()
    curve.animation_data.action = bpy.data.actions.new(name="Spin")
    
    fcurve = curve.animation_data.action.fcurves.new(data_path="rotation_euler", index=2)
    fcurve.keyframe_points.add(2)
    fcurve.keyframe_points[0].co = (0, 0)
    fcurve.keyframe_points[1].co = (100, 4 * math.pi)
