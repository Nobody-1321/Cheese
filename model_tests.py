import cv2  
import numpy as np
from tensorflow.keras import models # type: ignore

model = models.load_model('chess_model.h5')
imagen = cv2.imread("dataset_chess/p__78.png")
imagen = cv2.resize(imagen, (85, 85))
imagen = np.expand_dims(imagen, axis=0)
print(imagen.shape)
imagen = imagen/255

print( np.argmax(model.predict(imagen)) )
    