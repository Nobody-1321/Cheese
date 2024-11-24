import dearpygui.dearpygui as dpg
import tkinter as tk
import numpy as np
import cheese as che
import cv2

cap = che.capture_video(0, 1280, 720, 30)
drawing = False
roi_coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
i =0    

def select_texture(sender, app_data):
    global textures_data
    texture = app_data
    print(texture)
    dpg.set_value(textures_data[0]["id"], texture)

def update_texture():
    global cap, roi_coords,i

    che.update_texture_frame("camera_texture", che.get_frame(cap))
    che.update_roi_coords(roi_coords)
    che.update_texture_frame("camera_texture", che.draw_roi(che.get_texture("camera_texture")["frame"]
                                                            , roi_coords, (0, 255, 0), 2))
    
    #if i == 0:
    #    current_items = dpg.get_item_configuration("listbox_id")["items"]
    #    current_items.append("roi")
    #    dpg.configure_item("listbox_id", items=current_items)
    #    i = 1
    
        
    
    frame_data = che.image_to_texture(che.get_texture("camera_texture")["frame"])
    dpg.set_value(che.get_texture("camera_texture")["id"], frame_data)

def main_gui():
    global cap

    # Obtener las dimensiones de la pantalla usando Tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Cerrar la ventana raíz de Tkinter

    # Configuración de Dear PyGui
    dpg.create_context()
    
    dpg.create_viewport(title="Cheese", width=screen_width, height=screen_height)

    che.register_texture("camera_texture", 1280, 720, None)
    

    # Ventana para mostrar la cámara
    with dpg.window(label="Vista de Cámara", width=screen_width - 600, height=screen_height, no_close=True,
                    no_move=True, no_resize=True, no_title_bar=False, no_background=False, no_collapse=False,
                    no_scrollbar=True, pos=[0, 0], no_bring_to_front_on_focus=True):
        dpg.add_image(che.get_texture("camera_texture")["id"])

    # Ventana de opciones
    with dpg.window(label="Opciones", width=600, height=screen_height, no_close=True, no_move=True,
                    no_resize=True, no_title_bar=True, no_background=False, no_collapse=True,
                    no_scrollbar=False, pos=[screen_width - 600, 0]):
        
        dpg.add_text("Guardar ROI")
        dpg.add_button(label="Capturar", callback=che.save_roi_coords, tag="capture_button")
        dpg.add_button(label="Guardar", callback=che.save_roi_coords, tag="save_button")
        #mostrar informacion de las coordenadas de la roi
        dpg.add_text("Coordenadas de la ROI")
        #mostrar infomacion no con add_text sino con add_input_text
        
        dpg.add_input_text(label="x1", default_value="0", tag="x1")        
        dpg.add_input_text(label="y1", default_value="0", tag="y1")
        dpg.add_input_text(label="x2", default_value="0", tag="x2")
        dpg.add_input_text(label="y2", default_value="0", tag="y2")


        dpg.add_text("Texturas")
        dpg.add_listbox(label="Texturas", items=["camera_texture"], num_items=5, callback=select_texture, tag="listbox_id")



    # Configurar los eventos del mouse
    with dpg.handler_registry():
        dpg.add_mouse_release_handler(callback=che.mouse_release_callback)
        dpg.add_mouse_click_handler(callback=che.mouse_press_callback)
        dpg.add_mouse_drag_handler(callback=che.mouse_drag_callback)

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
