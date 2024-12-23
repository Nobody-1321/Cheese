import cv2
import numpy as np
import sklearn.cluster as cluster # type: ignore
#import matplotlib.pyplot as plt
from tensorflow.keras import layers, models # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping


def main():

    data_dir = './dataset_cheese/'
    object_labels = [0, 1, 2, 3]
    images_v = [f'{data_dir}cv__{j}.png' for j in range(0, 400)]
    images_p = [f'{data_dir}pb__{j}.png' for j in range(0, 400)]
    
    dataset_info = []  # Lista para guardar información completa de cada imagen
    features = []
    labels = []

    mat_images = []

    for i in range(0, 400):
        img = cv2.imread(images_v[i])
        img = img/255
        mat_images.append(img)
        features.append(img)
        labels.append(object_labels[0])

        img = cv2.imread(images_p[i])
        img = img/255
        mat_images.append(img)
        features.append(img)
        labels.append(object_labels[1])

    features = np.array(features)
    labels = np.array(labels)

    print(features.shape)    
    print(labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)



    model = models.Sequential()

# Primera capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)))  # 32 filtros, tamaño (3x3)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

# Segunda capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

# Tercera capa convolucional
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))

# Aplanar y conectar a una capa densa
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))  # Capa oculta
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))  # Capa de salida (3 clases)

# Resumen del modelo
    model.summary()

    model.compile(
    optimizer='adam',  # Método de optimización
    loss='sparse_categorical_crossentropy',  # Pérdida para clasificación multiclase
    metrics=['accuracy']  # Métrica principal
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Precisión en el conjunto de prueba: {test_acc * 100:.2f}%")

    model.save('chess_model.keras')

    print('Modelo guardado')

    model = models.load_model('chess_model.keras')
    imagen = cv2.imread("dataset_cheese/cv__62.png")
    imagen = cv2.resize(imagen, (200, 200))
    imagen = np.expand_dims(img, axis=0)
    print(imagen.shape)
    imagen = imagen/255

    print(model.predict(imagen))

if __name__ == '__main__':
    main()