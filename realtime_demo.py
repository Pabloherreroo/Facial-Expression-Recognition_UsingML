import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ============================================================
# 🧩 Clase necesaria para deserializar el modelo guardado
# ============================================================
class FeaturePipelineModel:
    def __init__(self, feature_extractor=None, scaler=None, pca=None, model=None):
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        self.pca = pca
        self.model = model

    def predict(self, X):
        X_scaled = self.scaler.transform(X) if self.scaler else X
        X_pca = self.pca.transform(X_scaled) if self.pca else X_scaled
        return self.model.predict(X_pca)

# ============================================================
# 🔍 HOG igual que en el entrenamiento
# ============================================================
def get_hog_features(pixels):
    image = np.array(pixels).reshape((48, 48))
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features

# ============================================================
# 🧠 Preprocesamiento de la cara
# ============================================================
def preprocess_face(face_img, model, scaler=None, pca=None, feature_type="hog"):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))

    if feature_type == "hog":
        features = get_hog_features(resized)
    else:
        features = resized.flatten()

    features = np.array(features).reshape(1, -1)

    if scaler:
        features = scaler.transform(features)
    if pca:
        features = pca.transform(features)

    # ✅ Ajuste silencioso de longitud
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features and features.shape[1] != expected_features:
        if features.shape[1] > expected_features:
            features = features[:, :expected_features]
        else:
            pad = np.zeros((1, expected_features - features.shape[1]))
            features = np.hstack([features, pad])

    return features

# ============================================================
# 🏷️ Emociones del dataset FER2013
# ============================================================
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ============================================================
# 🎯 Cargar modelo guardado
# ============================================================
print("🔄 Cargando modelo entrenado...")
bundle = joblib.load("best_emotion_model.pkl")

if isinstance(bundle, FeaturePipelineModel):
    model = bundle.model
    scaler = bundle.scaler
    pca = bundle.pca
    feature_type = getattr(bundle, "feature_extractor", "hog")
else:
    model = bundle
    scaler = None
    pca = None
    feature_type = "hog"

print("✅ Modelo cargado correctamente:")
print(f"   - Clasificador: {model.__class__.__name__}")
print(f"   - Feature type: {feature_type}")
print(f"   - PCA aplicada: {'Sí' if pca else 'No'}")

# ============================================================
# 📷 Iniciar cámara
# ============================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("\n🎥 Iniciando detección en tiempo real... (pulsa 'q' para salir)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        try:
            features = preprocess_face(face_roi, model, scaler, pca, feature_type)
            pred = model.predict(features)[0]
            emotion_label = EMOTIONS[int(pred)] if int(pred) < len(EMOTIONS) else "Unknown"
        except Exception as e:
            emotion_label = "Error"
            print(f"⚠️ Error procesando cara: {e}")

        # Dibujar recuadro y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Demo finalizada.")
