
def coordinate_cells(cells, reverse=False):

    if reverse:
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['8', '7', '6', '5', '4', '3', '2', '1']
    else:
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
      
    coordinates = {}
    for i, cell in enumerate(cells):
        letter = letters[i % 8]
        number = numbers[i // 8]
        coordinates[f'{letter}{number}'] = cell
    return coordinates