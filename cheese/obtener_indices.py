
def obtener_indices(columnas, filas): 
    indices = []
    for i in range(columnas):
        k1 = i * (filas + 1)
        k2 = k1 + filas + 1

        for j in range(filas):
           indices.append((k1 + j, k1 + j + 1, k2 + j, k2 + j + 1))

    return indices

#exportar la funcion
__all__ = ['obtener_indices']   