import re
import json
from collections import Counter
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def clean_text_russian_belarusian(text):
    cleaned = re.sub(r"[^а-яА-ЯёЁіІўЎ'\- ]+", ' ', text)
    cleaned_words = cleaned.split()
    
    transformed_words = []
    for word in cleaned_words:
        if word[0] == "ў":
            transformed_word = "у" + word[1:] 
            transformed_words.append(clean_word(transformed_word))
        elif word[0] == "Ў":
            transformed_word = "У" + word[1:] 
            transformed_words.append(clean_word(transformed_word))
        else:
            transformed_words.append(clean_word(word))

    return ' '.join(transformed_words)

def clean_text_english(text):
    cleaned = re.sub(r"[^a-zA-Z'\- ]+", ' ', text)
    cleaned_words = [clean_word(word) for word in cleaned.split()]
    return ' '.join(cleaned_words)

def clean_word(word):
    cleaned = word.strip("-,./*()' ")
    return cleaned.lower()

def create_frequency_dictionary(texts, language):
    all_text = ' '.join(texts)
    if language in ['russian', 'belarusian']:
        cleaned_text = clean_text_russian_belarusian(all_text)
    elif language == 'english':
        cleaned_text = clean_text_english(all_text)
    words = cleaned_text.split()
    frequency_dict = Counter(words)
    return frequency_dict

def save_dictionary(frequency_dict, language):
    sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))
    with open(f"{language}_frequency_dict.json", 'w', encoding='utf-8') as f:
        json.dump(sorted_dict, f, ensure_ascii=False)

def load_dictionary(language):
    try:
        with open(f"{language}_frequency_dict.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def read_texts_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

class FrequencyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frequency Dictionary")
        
        # Настройка меню выбора языка
        self.language_var = tk.StringVar(value="russian")
        language_menu = tk.Menu(self.root)
        self.root.config(menu=language_menu)
        language_menu.add_radiobutton(label="Russian", variable=self.language_var, value="russian", command=self.update_words)
        language_menu.add_radiobutton(label="Belarusian", variable=self.language_var, value="belarusian", command=self.update_words)
        language_menu.add_radiobutton(label="English", variable=self.language_var, value="english", command=self.update_words)

        # Кнопки сортировки
        sort_frame = tk.Frame(self.root)
        sort_frame.pack()
        tk.Button(sort_frame, text="Sort Alphabetically Ascending", command=self.sort_by_alphabet_ascending).pack(side=tk.LEFT)
        tk.Button(sort_frame, text="Sort Alphabetically Descending", command=self.sort_by_alphabet_descending).pack(side=tk.LEFT)

        frequency_sort_frame = tk.Frame(self.root)
        frequency_sort_frame.pack()
        tk.Button(frequency_sort_frame, text="Sort by Frequency Ascending", command=self.sort_by_frequency_ascending).pack(side=tk.LEFT)
        tk.Button(frequency_sort_frame, text="Sort by Frequency Descending", command=self.sort_by_frequency_descending).pack(side=tk.LEFT)

        # Поле для поиска слов
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.update_search_results)
        search_frame = tk.Frame(self.root)
        search_frame.pack()
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        tk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT)

        # Список для отображения слов
        self.word_listbox = tk.Listbox(self.root, width=50, height=15)
        self.word_listbox.pack()

        # Кнопки для редактирования слов
        edit_frame = tk.Frame(self.root)
        edit_frame.pack()
        tk.Button(edit_frame, text="Edit Word", command=self.edit_word).pack(side=tk.LEFT)
        tk.Button(edit_frame, text="Delete Word", command=self.delete_word).pack(side=tk.LEFT)
        tk.Button(edit_frame, text="Add New Word", command=self.add_word).pack(side=tk.LEFT)

        # Кнопка для добавления текста
        tk.Button(self.root, text="Add Texts", command=self.add_texts).pack(side=tk.BOTTOM)

        # Контейнер для новых кнопок
        analysis_frame = tk.Frame(self.root)
        analysis_frame.pack(side=tk.BOTTOM, pady=(10, 0))
        
        # Новые кнопки для расчета закона Ципфа и эмпирического закона Ципфа
        tk.Button(analysis_frame, text="Calculate Zipf's Law", command=self.calculate_zipf_law).pack(side=tk.TOP, pady=(5, 0))
        tk.Button(analysis_frame, text="Calculate Empirical Zipf's Law", command=self.calculate_empirical_zipf_law).pack(side=tk.TOP, pady=(5, 0))

        # Инициализация словаря частот
        self.frequency_dict = {}

    def update_words(self):
        language = self.language_var.get()
        self.frequency_dict = load_dictionary(language)
        self.show_words(self.frequency_dict.items())

    def update_search_results(self, *args):
        search_pattern = clean_word(self.search_var.get())
        matching_words = [(word, freq) for word, freq in self.frequency_dict.items() if word.startswith(search_pattern)]
        self.show_words(matching_words)

    def sort_by_alphabet_ascending(self):
        sorted_words = sorted(self.frequency_dict.items())
        self.show_words(sorted_words)

    def sort_by_alphabet_descending(self):
        sorted_words = sorted(self.frequency_dict.items(), reverse=True)
        self.show_words(sorted_words)

    def sort_by_frequency_ascending(self):
        sorted_words = sorted(self.frequency_dict.items(), key=lambda item: item[1])
        self.show_words(sorted_words)

    def sort_by_frequency_descending(self):
        sorted_words = sorted(self.frequency_dict.items(), key=lambda item: item[1], reverse=True)
        self.show_words(sorted_words)

    def show_words(self, words):
        self.word_listbox.delete(0, tk.END)
        for word, freq in words:
            self.word_listbox.insert(tk.END, f"{word}: {freq}")

    def edit_word(self):
        selected = self.word_listbox.get(tk.ACTIVE)
        if selected:
            word, _ = selected.split(": ")
            new_word = simpledialog.askstring("Edit Word", f"Edit the word '{word}':")
            if new_word:
                new_word = clean_word(new_word)  # Очищаем новое слово
                if new_word.lower() in self.frequency_dict:
                    self.frequency_dict[new_word.lower()] += self.frequency_dict[word]
                else:
                    self.frequency_dict[new_word.lower()] = self.frequency_dict[word]
                del self.frequency_dict[word]
                save_dictionary(self.frequency_dict, self.language_var.get())
                self.sort_by_frequency_descending()

    def delete_word(self):
        selected = self.word_listbox.get(tk.ACTIVE)
        if selected:
            word, _ = selected.split(": ")
            confirm = messagebox.askyesno("Delete Word", f"Are you sure you want to delete '{word}'?")
            if confirm:
                del self.frequency_dict[word]
                save_dictionary(self.frequency_dict, self.language_var.get())
                self.sort_by_frequency_descending()

    def add_word(self):
        new_word = simpledialog.askstring("Add New Word", "Enter the new word:")
        if new_word:
            new_word = clean_word(new_word)  # Очищаем новое слово
            new_word_lower = new_word.lower()
        
            if new_word_lower in self.frequency_dict:
                messagebox.showinfo("Word Exists", f"The word '{new_word}' already exists in the dictionary.")
            else:
                self.frequency_dict[new_word_lower] = 0
                save_dictionary(self.frequency_dict, self.language_var.get())
                self.sort_by_frequency_descending()

    def add_texts(self):
        filename = filedialog.askopenfilename()
        if filename:
            texts = read_texts_from_file(filename)
            language = self.language_var.get()
            frequency_dict = create_frequency_dictionary(texts, language)

            for word, freq in frequency_dict.items():
                if word in self.frequency_dict:
                    self.frequency_dict[word] += freq
                else:
                    self.frequency_dict[word] = freq

            save_dictionary(self.frequency_dict, language)
            self.show_words(self.frequency_dict.items())    
            
    def calculate_zipf_law(self):
        """Рассчитывает и отображает закон Ципфа, определяет коэффициент Ципфа и проверяет выполнение закона."""
        language = self.language_var.get()
        color_map = {
            "russian": "blue",
            "belarusian": "green",
            "english": "red"
        }
        color = color_map.get(language, "black")

        # Сортируем слова по частоте по убыванию
        sorted_words = sorted(self.frequency_dict.items(), key=lambda item: item[1], reverse=True)

        # Подготовка данных для анализа
        frequencies = [freq for _, freq in sorted_words]
        total_frequency = sum(frequencies)

        # Проверка на ненулевое значение
        if total_frequency == 0:
            print("Общая частота равна нулю. Проверьте данные.")
            return

        relative_frequencies = [freq / total_frequency for freq in frequencies]  # Относительные частоты
        ranks = range(1, len(relative_frequencies) + 1)  # Ранги от 1 до количества слов

        # Проверка выполнения закона Ципфа для всех слов
        c_values = [f * r for f, r in zip(relative_frequencies, ranks)]  # Вычисляем c для каждого слова
        average_c = sum(c_values) / len(c_values)  # Среднее значение c

        # Выводим результаты
        print(f"Среднее значение коэффициента Ципфа (c): {average_c:.4f}")

        # Построение графика только для первых 100 слов
        top_n = 100
        top_words = sorted_words[:top_n]
        top_frequencies = [freq for _, freq in top_words]
        top_relative_frequencies = [freq / total_frequency for freq in top_frequencies]
        top_ranks = range(1, len(top_relative_frequencies) + 1)

        # Вычисление f = c / r для графика
        f_values = [average_c / r for r in top_ranks]

        # График
        plt.figure()
        
        # График относительных частот
        plt.plot(top_ranks, top_relative_frequencies, marker=".", label="Относительная частота", color=color)
        
        # График функции f = c / r
        plt.plot(top_ranks, f_values, label="f = c / r", linestyle="--", color="orange")
        
        plt.xlabel("Ранг")
        plt.ylabel("Относительная частота / f = c / r")
        plt.title(f"Закон Ципфа ({language.capitalize()}): Топ-{top_n} слов")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def calculate_empirical_zipf_law(self):
        """Рассчитывает эмпирический закон Ципфа на основе длины слов и частоты для первых 100 слов."""
        # Задаем цвет графика в зависимости от языка
        language = self.language_var.get()
        color_map = {
            "russian": "blue",
            "belarusian": "green",
            "english": "red"
        }
        color = color_map.get(language, "black")  # Цвет для текущего языка, по умолчанию черный

        # Сортируем словарь частот и берем первые 100 слов
        sorted_words = sorted(self.frequency_dict.items(), key=lambda item: item[1], reverse=True)[:100]
        
        # Определяем длину каждого слова и частоты
        word_lengths = [len(word) for word, _ in sorted_words]
        frequencies = [freq for _, freq in sorted_words]  # Частоты первых 100 слов

        # Построение графика зависимости частоты от длины слова
        plt.figure()
        plt.scatter(word_lengths, frequencies, alpha=0.5, color=color)
        plt.xlabel("Длина слова")
        plt.ylabel("Частота")
        plt.title(f"Эмпирический закон Ципфа ({language.capitalize()}): Длина слова vs Частота (Топ-100 слов)")
        plt.show()
        
def calculate_juayan_coefficient(frequency_dict, language, texts):
    """Рассчитывает коэффициент Жуйана и сохраняет результаты в Excel, оптимизированная версия."""
    # Объединяем все тексты в одну строку
    all_text = ' '.join(texts)
    
    # Очищаем текст в зависимости от языка
    if language in ['russian', 'belarusian']:
        cleaned_text = clean_text_russian_belarusian(all_text)
    elif language == 'english':
        cleaned_text = clean_text_english(all_text)
    else:
        raise ValueError("Unsupported language")

    all_words = list(frequency_dict.keys())
    num_segments = 4
    segment_length = len(cleaned_text) // num_segments

    # Создаем сегменты на основе очищенного текста
    segments = [
        cleaned_text[i * segment_length: (i + 1) * segment_length] 
        for i in range(num_segments)
    ]
    # Добавляем оставшиеся слова в последний сегмент
    if len(cleaned_text) % num_segments != 0:
        segments[-1] += cleaned_text[num_segments * segment_length:]

    # Подсчет частот для каждого сегмента
    if language in ['russian', 'belarusian']:
        segment_frequencies = [
            Counter(clean_text_russian_belarusian(seg).split()) for seg in segments
        ]
    elif language == 'english':
        segment_frequencies = [
            Counter(clean_text_english(seg).split()) for seg in segments
        ]


    # Общая сумма для частоты слов
    total_frequency = sum(frequency_dict.values())

    # Предварительный расчет рангов слов
    sorted_words = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)
    rank_dict = {word: rank + 1 for rank, (word, _) in enumerate(sorted_words)}

    results = []

    for word in all_words:
        freq_in_segments = [seg_freq.get(word, 0) for seg_freq in segment_frequencies]
        n = sum(1 for freq in freq_in_segments if freq > 0)
        if n == 0:
            continue

        # Вычисляем μ — среднюю частоту
        mu = sum(freq_in_segments) / n
        if mu == 0:
            D = 0
        else:
            # Вычисляем σ — стандартное отклонение
            sigma = np.std(freq_in_segments, ddof=1)
            D = 100 * (1 - sigma / (mu * np.sqrt(n - 1))) if n > 1 else 0

        # Получаем относительную частоту и ранг
        relative_frequency = frequency_dict[word] / total_frequency
        rank = rank_dict[word]

        results.append((word, relative_frequency, rank, D, n))

    # Создание DataFrame и сохранение в Excel
    df = pd.DataFrame(results, columns=['Word', 'Relative Frequency', 'Rank', 'D', 'N'])
    excel_filename = f"{language}_juayan_coefficient_results_optimized.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"Результаты сохранены в файл: {excel_filename}")

# Основной блок для загрузки словарей и вызова функции
if __name__ == "__main__":
    # Загрузка словарей для каждого языка
    #russian_dict = load_dictionary('russian')
    #belarusian_dict = load_dictionary('belarusian')
    english_dict = load_dictionary('english')
    #russian_text = read_texts_from_file('russian_text.txt')  # Загрузите текстовый файл
    #belarusian_text = read_texts_from_file('belarusian_text.txt')
    english_text = read_texts_from_file('english_text.txt')

    # Вызываем функцию для каждого языка
    #calculate_juayan_coefficient(russian_dict, 'russian', russian_text)
    #calculate_juayan_coefficient(belarusian_dict, 'belarusian', belarusian_text)
    calculate_juayan_coefficient(english_dict, 'english', english_text)
    
    root = tk.Tk()
    app = FrequencyApp(root)
    root.mainloop()