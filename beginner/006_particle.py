"""
WIP: InteractiveOperator is not working.
"""
import bpy
import math
from mathutils import Vector

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

bpy.ops.mesh.primitive_plane_add(size=10)
plane = bpy.context.active_object
plane.modifiers.new("ParticalSystem", "PARTICLE_SYSTEM")
psys = plane.particle_systems[0].settings
psys.count = 200
psys.frame_start = 1
psys.frame_end = 1
psys.lifetime = 250
psys.physics_type = "NEWTON"


def add_force_field(location):
    bpy.ops.object.effector_add(type="FORCE", location=location)
    field = bpy.context.active_object
    field.field.strength = -500
    field.field.falloff_type = "SPHERE"
    pass


class InteractiveOperator(bpy.types.Operator):
    bl_idname = "object.physics_interaction"
    bl_label = "Physical Interaction"

    def modal(self, context, event):
        if not context.region_data:
            self.report({"ERROR"}, "Region data is None. Ensure the script is run in a 3D View.")
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y, 0))
            world_pos = context.region_data.view_matrix.inverted() @ mouse_pos
            add_force_field(mouse_pos)
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


bpy.utils.register_class(InteractiveOperator)
bpy.ops.object.physics_interaction("INVOKE_DEFAULT")
