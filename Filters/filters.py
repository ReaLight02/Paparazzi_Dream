import cv2


def dimensionCheck(
    eye1_center, img, overlay, w, h, offsetX=1, offsetY=1, musX=1, musY=1, glasses=True
):
    if w > 0 and h > 0:
        # resize the overaly
        overlay_resized = cv2.resize(overlay, (w, h))
        # upper left position
        x = int((eye1_center[0] - w // offsetX) * musX)
        if type:
            y = int((eye1_center[1] - h // offsetY) * musY)
        else:
            y = int((eye1_center[1] + h // offsetY) * musY)
        # check to make sure the position stays within the image
        x = max(x, 0)
        y = max(y, 0)
        if x + w > img.shape[1]:
            w = img.shape[1] - x
            overlay_resized = cv2.resize(overlay, (w, h))
        if y + h > img.shape[0]:
            h = img.shape[0] - y
            overlay_resized = cv2.resize(overlay, (w, h))

        # separate bgr and alpha channels
        b, g, r, a = cv2.split(overlay_resized)
        # apply glasses on the image
        img = applyFilter(img, overlay_resized, a, w, h, x, y)
        return img


def applyFilter(img, overlay, a, w, h, x, y):
    alpha = a / 255.0
    alpha_inv = 1.0 - alpha

    # apply the overlay on the image
    for c in range(0, 3):
        img[y : y + h, x : x + w, c] = (
            alpha * overlay[:, :, c] + alpha_inv * img[y : y + h, x : x + w, c]
        )
    return img


def Filters(img):
    # haar cascades
    face_cascade = cv2.CascadeClassifier("E:\LAB\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("E:\LAB\haarcascade_eye.xml")

    # load glasses and mustache with alpha channel
    glasses = cv2.imread("E:\LAB\glasses.png", cv2.IMREAD_UNCHANGED)
    mustache = cv2.imread("E:\LAB\mustache.png", cv2.IMREAD_UNCHANGED)

    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces in the image
    faces = face_cascade.detectMultiScale(
        gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:

        # cut the area around the face
        roi_gray = gray_face[y : y + h, x : x + w]

        # find the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            # Order the eyes and save the positions
            eyes = sorted(eyes, key=lambda eye: eye[0])
            eye1 = eyes[0]
            eye2 = eyes[1]

            # center of the eyes
            eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
            eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

            # size of the glasses
            eye_distance = eye2_center[0] - eye1_center[0]
            glasses_width = int(
                eye_distance * 2 * 1.4
            )  # the size of the glasses is two times the distance between the eyes made bigger by 30%
            scale_factor = glasses_width / glasses.shape[1]
            glasses_height = int(glasses.shape[0] * scale_factor)

            # apply the glasses on the face
            img = dimensionCheck(
                eye1_center, img, glasses, glasses_width, glasses_height, 3.5, 2
            )

            # find size of the mustache using the distance between the eyes
            mustache_width = int(eye_distance * 1.5)
            mustache_height = int(
                mustache.shape[0] * (mustache_width / mustache.shape[1])
            )

            # apply the mustache on the face
            img = dimensionCheck(
                eye1_center,
                img,
                mustache,
                mustache_width,
                mustache_height,
                4,
                4,
                1.03,
                1.35,
                False,
            )

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_image = cv2.imread(r"E:\LAB\smith.jpg")
    Filters(face_image)
