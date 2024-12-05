
def chess_coordinate_cells(cells, reverse=False):

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

def black_cells_chess_coordinates():
    black_cells = ['a1', 'c1', 'e1', 'g1',
                    'b2', 'd2', 'f2', 'h2',
                    'a3', 'c3', 'e3', 'g3',
                    'b4', 'd4', 'f4', 'h4',
                    'a5', 'c5', 'e5', 'g5',
                    'b6', 'd6', 'f6', 'h6',
                    'a7', 'c7', 'e7', 'g7',
                    'b8', 'd8', 'f8', 'h8']
    return black_cells 

def white_cells_chess_coordinates():
    white_cells = ['b1', 'd1', 'f1', 'h1',
                    'a2', 'c2', 'e2', 'g2',
                    'b3', 'd3', 'f3', 'h3',
                    'a4', 'c4', 'e4', 'g4',
                    'b5', 'd5', 'f5', 'h5',
                    'a6', 'c6', 'e6', 'g6',
                    'b7', 'd7', 'f7', 'h7',
                    'a8', 'c8', 'e8', 'g8']
    return white_cells

def all_chess_coordinates():
    chess_cells = [ 'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
                    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
                    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
                    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
                    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
                    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
                    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
                    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']
    return chess_cells