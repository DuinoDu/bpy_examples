"""
WIP: Texture is not showup in blender.
"""
import bpy
import requests
import io
import os
from PIL import Image
import tempfile

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


def generate_texture(prompt):
    url = "https://api.segmind.com/v1/ideogram-turbo-txt-2-img"
    api_key = os.getenv("SEGMIND_API_KEY")
    data = {
        "magic_prompt_option": "AUTO",
        "negative_prompt": "low quality,blurry",
        # "prompt": "portrait editorial, Img_4099.HEIC: cool anthropomorphic masculine tiger posing against neutral gray background, wearing alternative cyberpunk attire by Demna Gvasalia, rendered with photorealistic style, high definition, dof, create a fashion atmosphere, award winning photography, Hasselblad H6D-100c, XCD 85mm f/1.8",
        "prompt": prompt,
        "resolution": "RESOLUTION_1024_1024",
        "seed": 56698,
        "style_type": "GENERAL"
        }
    headers = {'x-api-key': api_key}
    print(f"Post to {url}")
    response = requests.post(url, json=data, headers=headers)
    print("Get response")
    image = Image.open(io.BytesIO(response.content)) 
    image.save(f"tmp_{prompt.replace(' ' , '_').png}")
    return pil_image_to_bpy_image(image)


def create_pbr_material(name, diffuse_map, normal_map):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    for node in nodes:
        nodes.remove(node)
    
    tex_diffuse = nodes.new("ShaderNodeTexImage")
    tex_diffuse.image = diffuse_map
    tex_normal = nodes.new("ShaderNodeTexImage")
    tex_normal.image = normal_map
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(tex_diffuse.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_normal.outputs['Color'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def pil_image_to_bpy_image(pil_image, name="NewImage"):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        pil_image.save(temp_file.name, 'PNG')
    bpy_image = bpy.data.images.load(temp_file.name)
    bpy_image.name = name
    bpy_image.pack()
    os.remove(temp_file.name)
    return bpy_image


diffuse_img = generate_texture("rusty metal texture")
normal_img = generate_texture("normal map of rusty metal")

bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, location=(0, 0, 0))
obj = bpy.context.active_object
obj.data.materials.append(create_pbr_material("AI Material", diffuse_img, normal_img))
