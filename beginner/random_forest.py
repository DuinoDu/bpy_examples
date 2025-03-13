import bpy
import random

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

for i in range(50):
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    height = random.uniform(0.5, 3)
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, height/2))
    obj = bpy.ops.object
    obj.scale = (0.5, 0.5, height)
    __import__('ipdb').set_trace()
    obj.data.materials.append(bpy.data.materials.new(name=f"Mat_{i}"))
