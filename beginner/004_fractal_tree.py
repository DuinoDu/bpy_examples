import bpy
import math
from mathutils import Vector, Euler

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

def create_branch(start, direction, length, depth, radius):
    if depth == 0:
        return

    location = start + direction * length / 2
    end = start + direction * length
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=location)
    branch = bpy.context.active_object
    branch.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    
    angle = Euler((0, math.radians(30), 0), 'XYZ')
    direction1 = direction.copy()
    direction1.rotate(angle)
    create_branch(end, direction1, length*0.6, depth-1, radius*0.8)
    angle = Euler((0, math.radians(-30), 0), 'XYZ')
    direction2 = direction.copy()
    direction2.rotate(angle)
    create_branch(end, direction2, length*0.6, depth-1, radius*0.8)

create_branch(Vector((0,0,0)), Vector((0,0,1)), 2, depth=5, radius=0.1)
