# Style-in-architecture
# Загрузка и предобработка изображений
head
def load_images(image_folder, img_size=(128, 128)):
    images = []
    labels = []
    for style_folder in os.listdir(image_folder):
        style_folder_path = os.path.join(image_folder, style_folder)
        if os.path.isdir(style_folder_path):
            for img_name in os.listdir(style_folder_path):
                img_path = os.path.join(style_folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(style_folder)
    return np.array(images), np.array(labels)

# Загрузка данных
image_folder = 'path_to_your_dataset'
X, y = load_images(image_folder)
# Нормализация изображений
X = X / 255.0
# Кодирование меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Разделение на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Построение модели нейронной сети
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Функция для предсказания стиля здания
def predict_style(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Пример использования
img_path = 'path_to_image.jpg'
predicted_style = predict_style(img_path)
print(f"Predicted Style: {predicted_style}")
