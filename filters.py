import cv2
import numpy as np

# Carica i classificatori Haar Cascade
face_cascade = cv2.CascadeClassifier('C:\\Users\\antop\\Desktop\\progetto\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\antop\\Desktop\\progetto\\haarcascade_eye.xml')

# Carica l'immagine del volto
face_image = cv2.imread("C:\\Users\\antop\\Desktop\\progetto\\Celebrity Faces Dataset\\Johnny Depp\\032_4b3ee537.jpg")

# Carica l'immagine degli occhiali con trasparenza (includendo il canale alpha)
glasses = cv2.imread("C:\\Users\\antop\\Desktop\\progetto\\cose\\glasses.png", cv2.IMREAD_UNCHANGED)
# Carica l'immagine dei baffi con trasparenza (includendo il canale alpha)
mustache = cv2.imread("C:\\Users\\antop\\Desktop\\progetto\\cose\\mustache.png", cv2.IMREAD_UNCHANGED)

# Converti l'immagine del volto in scala di grigi
gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

# Rileva i volti nell'immagine
faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Disegna il rettangolo attorno alla faccia rilevata
    cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Ritaglia l'area del volto rilevato
    roi_gray = gray_face[y:y + h, x:x + w]
    roi_color = face_image[y:y + h, x:x + w]

    # Rileva gli occhi nell'area del volto
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        # Disegna il rettangolo attorno agli occhi rilevati
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if len(eyes) >= 2:
        # Ordina gli occhi da sinistra a destra
        eyes = sorted(eyes, key=lambda eye: eye[0])

        # Prendi le coordinate dei due occhi
        eye1 = eyes[0]
        eye2 = eyes[1]

        # Calcola le coordinate centrali degli occhi relativi alla faccia
        eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
        eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

        # Calcola la larghezza e l'altezza degli occhiali
        eye_distance = eye2_center[0] - eye1_center[0]
        glasses_width = int(eye_distance * 2 * 1.3)  # Imposta la larghezza degli occhiali proporzionalmente alla distanza tra gli occhi, ingrandendo del 50%
        scale_factor = glasses_width / glasses.shape[1]
        glasses_height = int(glasses.shape[0] * scale_factor)

        # Assicurati che le dimensioni degli occhiali siano positive
        if glasses_width > 0 and glasses_height > 0:
            # Ridimensiona gli occhiali
            glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

            # Calcola la posizione superiore sinistra degli occhiali
            glasses_x = eye1_center[0] - glasses_width // 3.5
            glasses_y = eye1_center[1] - glasses_height // 2

            # Assicurati che la posizione non esca dall'immagine
            glasses_x = int(max(glasses_x, 0))
            glasses_y = max(glasses_y, 0)

            # Assicurati che gli occhiali non eccedano i bordi dell'immagine
            if glasses_x + glasses_width > face_image.shape[1]:
                glasses_width = face_image.shape[1] - glasses_x
                glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

            if glasses_y + glasses_height > face_image.shape[0]:
                glasses_height = face_image.shape[0] - glasses_y
                glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

            # Separare i canali BGR e Alpha degli occhiali ridimensionati
            b, g, r, a = cv2.split(glasses_resized)
            alpha = a / 255.0
            alpha_inv = 1.0 - alpha

            # Applicare gli occhiali con trasparenza sull'immagine del volto
            for c in range(0, 3):
                face_image[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width, c] = (
                    alpha * glasses_resized[:, :, c] +
                    alpha_inv * face_image[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width, c]
                )

        # Calcola la larghezza e l'altezza dei baffi
        mustache_width = int(eye_distance * 1.5)  # Imposta la larghezza dei baffi proporzionalmente alla distanza tra gli occhi
        mustache_height = int(mustache.shape[0] * (mustache_width / mustache.shape[1]))

        # Assicurati che le dimensioni dei baffi siano positive
        if mustache_width > 0 and mustache_height > 0:
            # Ridimensiona i baffi
            mustache_resized = cv2.resize(mustache, (mustache_width, mustache_height))

            # Calcola la posizione superiore sinistra dei baffi
            mustache_x = int((eye1_center[0] - mustache_width // 4)*1.1)
            mustache_y = int((eye2_center[1] + eye_distance // 4)*1.2)

            # Assicurati che la posizione non esca dall'immagine
            mustache_x = max(mustache_x, 0)
            mustache_y = max(mustache_y, 0)

            # Assicurati che i baffi non eccedano i bordi dell'immagine
            if mustache_x + mustache_width > face_image.shape[1]:
                mustache_width = face_image.shape[1] - mustache_x
                mustache_resized = cv2.resize(mustache, (mustache_width, mustache_height))

            if mustache_y + mustache_height > face_image.shape[0]:
                mustache_height = face_image.shape[0] - mustache_y
                mustache_resized = cv2.resize(mustache, (mustache_width, mustache_height))

            # Separare i canali BGR e Alpha dei baffi ridimensionati
            b, g, r, a = cv2.split(mustache_resized)
            alpha = a / 255.0
            alpha_inv = 1.0 - alpha

            # Applicare i baffi con trasparenza sull'immagine del volto
            for c in range(0, 3):
                face_image[mustache_y:mustache_y + mustache_height, mustache_x:mustache_x + mustache_width, c] = (
                    alpha * mustache_resized[:, :, c] +
                    alpha_inv * face_image[mustache_y:mustache_y + mustache_height, mustache_x:mustache_x + mustache_width, c]
                )

# Mostra l'immagine risultante
cv2.imshow('Result', face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()