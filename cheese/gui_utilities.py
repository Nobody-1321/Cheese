import dearpygui.dearpygui as dpg
import numpy as np


def create_dynamic_texture(tag, width, height):
    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width, height,
            np.zeros((height, width, 4), dtype=np.float32),
            tag=tag
        )
        
    return tag