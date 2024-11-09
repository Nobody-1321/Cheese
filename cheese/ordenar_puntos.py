

def ordenar_puntos(puntos):
    puntos_ordenados_x = sorted(puntos, key=lambda punto: (punto[1], punto[0]))
    
    puntos_interseccion_ordenados2 = []

    #ordenar solo en y
    for i in range(0, len(puntos_ordenados_x), 9):
        fila = puntos_ordenados_x[i:i+9]  # Seleccionar 9 puntos
        fila_ordenada = sorted(fila, key=lambda punto: punto[0])  # Ordenar por la coordenada x
        puntos_interseccion_ordenados2.extend(fila_ordenada) 

    return puntos_interseccion_ordenados2

__all__ = ['ordenar_puntos']