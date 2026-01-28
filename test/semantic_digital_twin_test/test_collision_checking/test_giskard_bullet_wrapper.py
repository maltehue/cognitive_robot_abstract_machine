import os

from pkg_resources import resource_filename

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.collision_checking.bpb_wrapper import (
    clear_cache,
    convert_to_decomposed_obj_and_save_in_tmp,
)
from semantic_digital_twin.world_description.world_entity import Body


def test_vhacd():
    stl_path = os.path.join(
        resource_filename("semantic_digital_twin", "../../"),
        "resources",
        "stl",
        "jeroen_cup.stl",
    )
    world_with_stl = STLParser(stl_path).parse()
    body: Body = world_with_stl.root
    mesh = body.collision[0]
    clear_cache()
    convert_to_decomposed_obj_and_save_in_tmp(mesh.mesh)
