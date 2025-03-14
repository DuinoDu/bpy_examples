import bpy
import csv

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

csv_path = "./data.csv"
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        height = float(row["value"])
        bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=height, location=(i*2, 0, height/2))
        obj = bpy.context.active_object
        obj.name = f"Bar_{row['name']}"
        mat = bpy.data.materials.new(name=f"Mat_{row['name']}")
        mat.diffuse_color = (i * 0.3, 0.5, 0.8, 1)
        obj.data.materials.append(mat)
