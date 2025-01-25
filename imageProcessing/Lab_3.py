import numpy as np
import cv2 as cv
import PIL
from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image
from copy import deepcopy


class MainSolution():
    def __init__(self):                                  #конструктор класса MainSolution
        self.image = cv.imread("images/rice.png")      #загружаем изображение
        self.imgray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) #преобразуем в серый
        self.trsh1 = None
        self.trsh2 = None                               #атрибуты для хранения пороговых значений

    def original(self):  #оригинальное изображение
        img = Image.fromarray(cv.cvtColor(self.image, cv.COLOR_BGR2RGB)) #из BGR в RGB. Создается объект Image с использованием массива пикселей изображения
        img = img.resize((200, 200))     #устанавливаем размер
        return ImageTk.PhotoImage(img)

    def adaptive_threshold(self):
        thresh2 = cv.adaptiveThreshold(self.imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        self.trsh2 = deepcopy(thresh2)
        img = Image.fromarray(thresh2)
        img = img.resize((200, 200))
        return ImageTk.PhotoImage(img)
    
    def laplacian(self):
        # Применение фильтра Лапласа
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        laplacian_image = cv.Laplacian(gray_image, cv.CV_64F)
        laplacian_image = cv.convertScaleAbs(laplacian_image)
        laplacian_image = cv.cvtColor(laplacian_image, cv.COLOR_GRAY2RGB)
        img = Image.fromarray(laplacian_image)
        img = img.resize((200, 200))
        return ImageTk.PhotoImage(img)
    
    def laplacian_of_gaussian(self):
        # Применение фильтра Лапласа-Гаусса
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)
        log_image = cv.Laplacian(blurred_image, cv.CV_64F)
        log_image = cv.convertScaleAbs(log_image)
        log_image = cv.cvtColor(log_image, cv.COLOR_GRAY2RGB)
        img = Image.fromarray(log_image)
        img = img.resize((200, 200))
        return ImageTk.PhotoImage(img)
    
    def bernsen_threshold(self, window_size=15, contrast_threshold=15):
        # Применение метода Бернсена
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        binary_image = np.zeros_like(gray_image)

        height, width = gray_image.shape

        half_window = window_size // 2

        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                window = gray_image[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
                min_val = np.min(window)
                max_val = np.max(window)
                contrast = max_val - min_val

                if contrast <= contrast_threshold:
                    binary_image[i, j] = 255 if gray_image[i, j] > ((min_val + max_val) // 2) else 0
                else:
                    binary_image[i, j] = 255 if gray_image[i, j] > ((min_val + max_val) // 2) else 0

        img = Image.fromarray(binary_image)
        img = img.resize((200, 200))
        return ImageTk.PhotoImage(img)
    
    def niblack_threshold(self, window_size=15, k=-0.2):
        # Применение метода Ниблэка
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        binary_image = np.zeros_like(gray_image)

        height, width = gray_image.shape

        half_window = window_size // 2

        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                window = gray_image[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
                mean = np.mean(window)
                std = np.std(window)
                threshold = mean + k * std

                binary_image[i, j] = 255 if gray_image[i, j] > threshold else 0

        img = Image.fromarray(binary_image)
        img = img.resize((200, 200))
        return ImageTk.PhotoImage(img)



if __name__ == "__main__":
    root = Tk()
    ms = MainSolution()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"900x600") 

    methods = [
        ms.original,
        ms.bernsen_threshold,
        ms.niblack_threshold,
        ms.adaptive_threshold,
        ms.laplacian,
        ms.laplacian_of_gaussian
    ]

    method_labels = [
        "Оригинал",
        "Локальная пороговая обработка (Бернсена)",
        "Локальная пороговая обработка (Ниблэка)",
        "Адаптивная пороговая обработка",
        "Фильтр Лапласса",
        "Фильтр Лапласса-Гаусса"
    ]

    img_width, img_height = 250, 200
    row1, row2 = 3, 6  # Количество методов на верхней и нижней строке

    # Верхняя часть окна
    for i in range(row1):
        img = methods[i]()
        lbl = ttk.Label(image=img)
        lbl.image = img
        lbl.place(x=i * (img_width + 20) + 30, y=40, width=img_width, height=img_height)
        lbl_text = ttk.Label(text=method_labels[i])
        lbl_text.place(x=i * (img_width + 20) + 30, y=250)

    # Нижняя часть окна
    for i in range(row1, row2):
        img = methods[i]()
        lbl = ttk.Label(image=img)
        lbl.image = img
        lbl.place(x=(i - row1) * (img_width + 20) + 30, y=350, width=img_width, height=img_height)  # Изменяем расположение
        lbl_text = ttk.Label(text=method_labels[i])
        lbl_text.place(x=(i - row1) * (img_width + 20) + 30, y=560)  # Изменяем расположение

    root.mainloop()