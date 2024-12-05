import cheese as che

pat = 'img/tableroA.png'

image = che.open_image(pat)

if image is None:
    print(f'Error: Could not open image {pat}')
    exit()
else:
    print(f'Image {pat} opened successfully')


gray_image = che.convert_to_gray(image)