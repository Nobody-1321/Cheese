import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report

def main():
    data_dir = '/home/magno1321/Documentos/cheese/dataset_cheesee/'
    #data_dir = './dataset_cheese/'
    object_labels = [0, 1]
    
    images_v = [f'{data_dir}v__{j}.png' for j in range(260, 290)]
    images_p = [f'{data_dir}tb__{j}.png' for j in range(260, 290)]
    
    features = []
    labels = []

    for i in range(0, 30):
        # Clase 0
        img = cv2.imread(images_v[i])
        img = img / 255.0
        features.append(img)
        labels.append(object_labels[0])

        # Clase 1
        img = cv2.imread(images_p[i])
        img = img / 255
        features.append(img)
        labels.append(object_labels[1])

    features = np.array(features)
    labels = np.array(labels)

    # Cargar el modelo
    model = models.load_model('chess_model.keras')

    # Realizar predicciones
    predictions = model.predict(features, verbose=1)
    
    # Mostrar las predicciones en porcentaje
    for predic in predictions:
        print(predic * 100)

    predicted_classes = np.argmax(predictions, axis=1)

    # Calcular la matriz de confusión
    cm = confusion_matrix(labels, predicted_classes)

    # Mostrar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.show()

    # Calcular métricas adicionales

    # 1. Precisión (Accuracy)
    accuracy = accuracy_score(labels, predicted_classes)
    print(f"Precisión (Accuracy): {accuracy * 100:.2f}%")

    # 2. Precisión (Precision)
    precision = precision_score(labels, predicted_classes)
    print(f"Precisión: {precision * 100:.2f}%")

    # 3. Recuperación (Recall)
    recall = recall_score(labels, predicted_classes)
    print(f"Recuperación (Recall): {recall * 100:.2f}%")

    # 4. F1-Score
    f1 = f1_score(labels, predicted_classes)
    print(f"F1-Score: {f1 * 100:.2f}%")

    # 5. Reporte de clasificación (detallado por cada clase)
    report = classification_report(labels, predicted_classes)
    print("Reporte de Clasificación:\n", report)

    # 6. Matriz de Confusión Normalizada
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, 
                                             display_labels=['Class 0', 'Class 1'])
    disp_normalized.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión Normalizada")
    plt.show()

    # 7. Curva ROC y AUC
    fpr, tpr, _ = roc_curve(labels, predictions[:, 1])  # Asumiendo que el modelo tiene 2 clases
    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
