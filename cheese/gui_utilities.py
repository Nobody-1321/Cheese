import time
import dearpygui.dearpygui as dpg
import numpy as np


textures_data = []

pressed = False
dragged = False
drawing = False


x_pos = 0
y_pos = 0


def create_dynamic_texture(tag, width, height):
    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width, height,
            np.zeros((height, width, 4), dtype=np.float32),
            tag=tag
        )        
    return tag

def mouse_release_callback(sender, app_data):
    global pressed
    pressed = False
    
def mouse_drag_callback(sender, app_data):
    global dragged, x_pos, y_pos
    _, x_pos_, y_pos_ = app_data

    if x_pos != x_pos_ or y_pos != y_pos_:
        dragged = True
    else:
        dragged = False

    _, x_pos, y_pos = app_data
    
def mouse_press_callback(sender, app_data):
    global pressed
    pressed = True

def mouse_right_button_state():
    return pressed, dragged

def mouse_position():
    x, y = dpg.get_mouse_pos()
    return x, y

def update_roi_coords(roi_coords):
    x, y = dpg.get_mouse_pos()
    global pressed, dragged, drawing

    if pressed and not drawing:
        drawing = True
        roi_coords["x1"], roi_coords["y1"] = x, y
         
    if dragged and drawing:
        roi_coords["x2"], roi_coords["y2"] = x, y
    
    if not pressed and drawing:
        drawing = False
        roi_coords["x2"], roi_coords["y2"] = x, y

    print(roi_coords)

def save_roi_coords(sender, app_data):
    pass

def reset_roi_coords(sender, app_data):
    roi_coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

def register_texture(tag, width, height, frame):
    global textures_data
    textures_data.append({"id": create_dynamic_texture(tag, width, height), "width": width, "height": height, "frame": frame})

def get_texture(tag):
    global textures_data
    for texture in textures_data:
        if texture["id"] == tag:
            return texture
    return None

def update_texture_frame(tag, frame):
    global textures_data
    texture = get_texture(tag)
    texture["frame"] = frame