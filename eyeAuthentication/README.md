# Система аутентификации по радужной оболочке глаза

Данная программа реализует систему аутентификации сотрудников компании на основе анализа изображений радужной оболочки глаза. Она использует методы обработки изображений для извлечения признаков и сравнения их с данными в базе.

## Установка

### Необходимые библиотеки

Для работы программы вам понадобятся следующие библиотеки:

- OpenCV
- NumPy
- Scikit-image
- Scikit-learn
- Flask
- Cryptography

Вы можете установить их с помощью pip:
pip install opencv-python numpy scikit-image scikit-learn flask cryptography

## Использование

### Импорт необходимых библиотек 
Убедитесь, что все необходимые библиотеки импортированы в вашем коде.
### Создание базы данных: 
Если файл employees.json отсутствует, программа автоматически создаст базу данных с примерами пользователей. В противном случае, вы можете загрузить существующую базу данных.
### Запуск Flask-приложения:
Для запуска приложения выполните следующую команду:
python your_script.py
После этого приложение будет доступно по адресу http://127.0.0.1:5000.
### Аутентификация:
Чтобы аутентифицироваться, отправьте POST-запрос на /authenticate с изображением в формате файла. Пример запроса с использованием curl: 
curl -X POST -F "image=@path_to_your_image.jpg" http://127.0.0.1:5000/authenticate
В случае успешной аутентификации вы получите ответ с данными пользователя.

## Структура программы

- Предобработка изображений: Модуль для чтения и нормализации изображений.
- Извлечение признаков: Модуль, использующий Габоровы фильтры для извлечения признаков изображения.
- Сравнение: Модуль для сравнения входного изображения с базой данных пользователей.
- Работа с базой данных: Модули для загрузки и сохранения базы данных пользователей.
- Безопасность данных: Шифрование и дешифрование базы данных для защиты данных пользователей.
