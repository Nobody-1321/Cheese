import dearpygui.dearpygui as dpg
import tkinter as tk
import numpy as np
import cheese as che
import cv2

textures_id = []
cap = che.capture_video(0, 1280, 720, 30)
drawing = False
roi_coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

def update_texture():
    global textures_id, cap
    frame_data = che.get_frame_tex(cap)
    dpg.set_value(textures_id[0]["id"], frame_data)

    if drawing or (roi_coords["x1"] != roi_coords["x2"] and roi_coords["y1"] != roi_coords["y2"]):
        x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]
        

def mouse_release_callback(sender, app_data):
    global drawing
    drawing = False
    print("Mouse released")

def mouse_press_callback(sender, app_data):
    global roi_coords, drawing
    mouse_x, mouse_y = dpg.get_mouse_pos()
    roi_coords["x1"], roi_coords["y1"] = mouse_x, mouse_y
    drawing = True
    print("Mouse pressed")

def draw_rectangle_mouse_callback(sender, app_data):
    global roi_coords, drawing
    mouse_x, mouse_y = dpg.get_mouse_pos()

    if drawing:
        roi_coords["x2"], roi_coords["y2"] = mouse_x, mouse_y
        update_texture()
    

def main_gui():
    global textures_id    

    # Obtener las dimensiones de la pantalla usando Tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Cerrar la ventana raíz de Tkinter

    # Configuración de Dear PyGui
    dpg.create_context()
    
    dpg.create_viewport(title="Cheese", width=screen_width, height=screen_height)

    textures_id.append({"id": che.create_dynamic_texture("camera_texture", 1280, 720), "width": 1280, "height": 720})
    
    # Ventana para mostrar la cámara
    with dpg.window(label="Vista de Cámara", width=screen_width - 600, height=screen_height, no_close=True,
                    no_move=True, no_resize=True, no_title_bar=False, no_background=False, no_collapse=False,
                    no_scrollbar=True, pos=[0, 0], no_bring_to_front_on_focus=True):
        dpg.add_image(textures_id[0]["id"])

    # Ventana de opciones
    with dpg.window(label="Opciones", width=600, height=screen_height, no_close=True, no_move=True,
                    no_resize=True, no_title_bar=True, no_background=False, no_collapse=True,
                    no_scrollbar=False, pos=[screen_width - 600, 0]):
        dpg.add_text("Ajustes de la cámara")
        dpg.add_slider_float(label="Brillo", default_value=1.0, min_value=0.0, max_value=2.0, tag="Brillo")
        dpg.add_slider_float(label="Contraste", default_value=1.0, min_value=0.0, max_value=2.0, tag="Contraste")
        dpg.add_slider_float(label="Saturación", default_value=1.0, min_value=0.0, max_value=2.0, tag="Saturación")
        dpg.add_text("Ajustes de la cámara")

    # Configurar los eventos del mouse
    with dpg.handler_registry():
        dpg.add_mouse_release_handler(callback=mouse_release_callback)
        dpg.add_mouse_click_handler(callback=mouse_press_callback)
        dpg.add_mouse_drag_handler(callback=draw_rectangle_mouse_callback)
        
    dpg.setup_dearpygui()
    dpg.show_viewport()

    #evitara que la ventana principal se redimensione
    dpg.set_viewport_resizable(False)

    # Inicializar la cámara

    # Bucle principal de la interfaz
    while dpg.is_dearpygui_running():
        update_texture()
        dpg.render_dearpygui_frame()

    # Liberar recursos al cerrar
    cap.release()
    dpg.destroy_context()


if __name__ == '__main__':
    main_gui()
