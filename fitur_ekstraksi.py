import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# Warna label (RGB untuk OpenCV)
warna_label = {
    "unripe": (0, 255, 0),       # hijau
    "ripe": (255, 255, 0),       # kuning
    "overripe": (165, 42, 42)    # coklat
}

# Ekstraksi fitur gabungan: hue histogram, statistik warna, GLCM
def ekstrak_fitur_gabungan(pil_img):
    img = pil_img.resize((64, 64))
    img_np = np.array(img.convert('RGB'))

    # HSV Histogram (Hue only)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    hist_hue = cv2.calcHist([hue], [0], None, [36], [0, 180])
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()

    # Statistik warna (hue)
    mean = np.mean(hue)
    std = np.std(hue)
    skew = ((hue - mean)**3).mean() / (std**3 + 1e-6)
    kurt = ((hue - mean)**4).mean() / (std**4 + 1e-6)
    stats = [mean, std, skew, kurt]

    # GLCM fitur
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    glcm_feats = [graycoprops(glcm, prop)[0, 0] for prop in glcm_props]

    return np.concatenate([hist_hue, stats, glcm_feats])

# Fungsi visualisasi hasil deteksi + prediksi per buah
def visualisasi_prediksi(gambar_np, boxes, labels, scores):
    gambar_np = gambar_np.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        warna = warna_label.get(label, (255, 255, 255))
        cv2.rectangle(gambar_np, (x1, y1), (x2, y2), warna, 2)
        teks = f"{label} ({score:.2f})"
        cv2.putText(gambar_np, teks, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)
    return gambar_np
