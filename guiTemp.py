import dearpygui.dearpygui as dpg

# Crear dos contextos
context_1 = dpg.create_context()
context_2 = dpg.create_context()

# Activar el primer contexto
dpg.set_context(context_1)
with dpg.window(label="Ventana del Contexto 1"):
    dpg.add_text("Este es el contexto 1.")

# Activar el segundo contexto
dpg.set_context(context_2)
with dpg.window(label="Ventana del Contexto 2"):
    dpg.add_text("Este es el contexto 2.")

# Mostrar el contexto 1
dpg.set_context(context_1)
dpg.create_viewport(title="Ejemplo Contexto 1", width=400, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()

# Alternar entre contextos durante el bucle
while dpg.is_dearpygui_running():
    dpg.set_context(context_1)
    dpg.render_dearpygui_frame()

    dpg.set_context(context_2)
    dpg.render_dearpygui_frame()

# Liberar recursos de ambos contextos
dpg.set_context(context_1)
dpg.destroy_context()
dpg.set_context(context_2)
dpg.destroy_context()
