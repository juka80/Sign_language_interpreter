
import cv2
import numpy as np

video_path =  cv2.VideoCapture("videos")

def preprocessing(video_path):
    """Przetwarza obraz z pliku wideo stosując redukcję szumów i poprawę kontrastu.
    :param video_path: Ścieżka do pliku wideo do przetworzenia.
    :type video_path: str
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie udało się otworzyć pliku wideo. Sprawdź ścieżkę.")
        exit()

    while True:
        ret, frame = cap.read()

        if ret:
            median_filtered = cv2.medianBlur(frame, 5)
            gaussian_filtered = cv2.GaussianBlur(frame, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            if len(frame.shape) == 3: 
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_clahe = clahe.apply(l)
                lab_clahe = cv2.merge((l_clahe, a, b))
                clahe_filtered = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            else:
                clahe_filtered = clahe.apply(frame)

            yield clahe_filtered
        else:
            break

    cap.release()

def segmentation(frame, use_canny=False, threshold1=100, threshold2=200):
    """Segmentuje rękę na obrazie stosując wybraną metodę segmentacji.

    :param frame: Ramka wideo do segmentacji.
    :type frame: numpy.ndarray
    :param use_canny: Jeśli True, używa detekcji krawędzi Canny, w przeciwnym razie stosuje binaryzację.
    :type use_canny: bool
    :param threshold1: Pierwszy próg dla operatora Canny.
    :type threshold1: int
    :param threshold2: Drugi próg dla operatora Canny.
    :type threshold2: int
    :return: Obraz z segmentowaną ręką.
    :rtype: numpy.ndarray
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if use_canny:
        # Użycie detekcji krawędzi Canny
        edged = cv2.Canny(gray, threshold1, threshold2)
        return edged
    else:
        # Binaryzacja obrazu
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary

#wtf?
    # Przykładowe użycie
# frame = ...  # wczytaj ramkę z funkcji preprocessing
# segmented_frame = segment_hand(frame, use_canny=False)

def pca_to_frame(frame, n_components=50):

    """Stosuje PCA do pojedynczej ramki w celu redukcji wymiarowości i ekstrakcji cech.

    :param frame: Ramka wideo do przetworzenia.
    :type frame: numpy.ndarray
    :param n_components: Liczba głównych składowych do zachowania.
    :type n_components: int
    :return: Ramka po redukcji wymiarowości.
    :rtype: numpy.ndarray
    """

    flat_frame = frame.flatten().reshape(1, -1)
    pca = PCA(n_components=n_components)
    pca_frame = pca.fit_transform(flat_frame)

    return pca_frame