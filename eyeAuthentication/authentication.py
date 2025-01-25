# Импорт необходимых библиотек
import cv2
import numpy as np
from skimage.filters import gabor
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet

# --- Модуль: Предобработка изображений ---
def preprocess_image(image_path):
    # Чтение изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The image was not found or damaged.")
    # Нормализация изображения
    image = cv2.resize(image, (128, 128))
    return image

# --- Модуль: Извлечение признаков ---
def extract_features(image):
    # Применение Габоровых фильтров
    features = []
    for theta in range(4):
        theta_rad = theta * np.pi / 4
        real, _ = gabor(image, frequency=0.6, theta=theta_rad)
        features.append(real.mean())
    return np.array(features)

# --- Модуль: Сравнение ---
def recognize_user(image, db):
    # Предобработка изображения
    processed_image = preprocess_image(image)

    # Извлечение признаков
    input_vector = extract_features(processed_image)
    
    # Поиск совпадений в базе данных
    for user in db['employees']:
        db_vector = np.array(user['vector'])
        similarity = cosine_similarity([input_vector], [db_vector])[0][0]
        print(f"Comparing with user: {user['name']}, Similarity: {similarity}, DB Vector: {db_vector}")
        
        if similarity > 0.999:  # Пороговое значение
            return user
    return None


# --- Модуль: Работа с базой данных ---
def save_database(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_database(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Пример создания базы данных с генерацией векторов
def create_sample_database():
    database = {
        "employees": [
            {"name": "Ivan Ivanov", "id": "001", "image": "sample_image_1.jpg"},
            {"name": "Maria Petrova", "id": "002", "image": "sample_image_2.jpg"}
        ]
    }
    
    # Генерация векторов для каждого пользователя
    for user in database["employees"]:
        image_path = user["image"]
        processed_image = preprocess_image(image_path)
        feature_vector = extract_features(processed_image)
        user["vector"] = feature_vector.tolist()  # Сохраняем вектор в базе данных
    
    # Сохраняем базу данных
    save_database(database, "employees.json")

# --- Модуль: Безопасность данных ---
def encrypt_database(data, key):
    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(json.dumps(data).encode())

def decrypt_database(encrypted_data, key):
    cipher_suite = Fernet(key)
    return json.loads(cipher_suite.decrypt(encrypted_data).decode())

# --- Модуль: Управляющая программа ---
app = Flask(__name__)

# Генерация ключа для шифрования
key = Fernet.generate_key()

# Загрузка базы данных
try:
    db = load_database("employees.json")
except FileNotFoundError:
    create_sample_database()
    db = load_database("employees.json")

@app.route("/authenticate", methods=["POST"])
def authenticate():
    # Получение изображения из POST-запроса
    image = request.files['image']
    if not image:
        return jsonify({"status": "fail", "message": "No image provided"}), 400
    try:
        # Временное сохранение изображения
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        user_data = recognize_user(temp_path, db)
        if user_data:
            return jsonify({"status": "success", "user": user_data}), 200
        else:
            return jsonify({"status": "fail", "message": "Access is denied"}), 403
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Запуск Flask-приложения ---
if __name__ == "__main__":
    app.run(debug=True)
