import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# Estrazione delle feature dal dataset di addestramento
def extract_features(generator, model):
    features = []
    labels = []
    
    for i in range(len(generator)):
        x, y = next(generator)
        feature = model.predict(x)
        features.append(feature)
        labels.append(y)
        
    return np.vstack(features), np.vstack(labels)

# Creazione di un nuovo modello che tronca l'ultimo strato softmax
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Estrazione delle feature dal dataset di addestramento e validazione
train_features, train_labels = extract_features(train_generator, feature_extractor)
val_features, val_labels = extract_features(val_generator, feature_extractor)

# Codifica delle etichette
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(np.argmax(train_labels, axis=1))
val_labels_enc = label_encoder.transform(np.argmax(val_labels, axis=1))

# Addestramento di un SVM sui feature estratti
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(train_features, train_labels_enc)

# Valutazione del classificatore
val_predictions = svm_classifier.predict(val_features)
accuracy = np.mean(val_predictions == val_labels_enc)
print(f'Validation Accuracy: {accuracy}')
