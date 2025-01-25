import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import io
import sys
import stanza

# Установка кодировки консоли на UTF-8 (для Windows)
os.system('chcp 65001')

# Установка кодировки вывода
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Загрузка моделей spaCy и Stanza
nlp_en = spacy.load('en_core_web_sm')  # Английский
nlp_ru = spacy.load('ru_core_news_sm')  # Русский
nlp_stanza_be = stanza.Pipeline('be', processors='tokenize,pos,lemma')

custom_stopwords_be = {"і", "ў", "на", "з", "што", "гэта", "гэтак", "для", "уся", "быў"}

def lemmatize_text(text, language):
    if language == 'be':
        doc = nlp_stanza_be(text)
        return ' '.join([word.lemma for sentence in doc.sentences for word in sentence.words if word.lemma not in custom_stopwords_be])
    else:
        nlp = {'en': nlp_en, 'ru': nlp_ru}.get(language)
        if not nlp:
            raise ValueError(f"Unsupported language: {language}")
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Функция для нормализации текста
def preprocess_text(text, language):
    return lemmatize_text(text, language)

# Функция для обработки одного языка
def process_language(language, texts):
    # Нормализация текста
    texts_normalized = [preprocess_text(sent, language) for sent in texts]

    # Разделение на первые и вторые 5 предложений
    part1 = texts_normalized[:5]
    part2 = texts_normalized[5:]

    # TF-IDF для первых 5 предложений
    vectorizer1 = TfidfVectorizer()
    tfidf_part1 = vectorizer1.fit_transform(part1)
    headers_part1 = vectorizer1.get_feature_names_out()
    
    print(f"\n--- TF и IDF для первых 5 предложений ({language}) ---")
    for i, sent in enumerate(part1):
        print(f"Предложение {i+1}: {sent}")
        tf_values = tfidf_part1[i].toarray()[0]
        for word, tf in zip(headers_part1, tf_values):
            print(f"  TF[{word}] = {tf:.4f}")
    for word, idf in zip(headers_part1, vectorizer1.idf_):
        print(f"IDF[{word}] = {idf:.4f}")

    df_tfidf_part1 = pd.DataFrame(tfidf_part1.toarray(), columns=headers_part1)
    df_tfidf_part1.to_excel(f"{language}_tfidf_part1.xlsx", index=False)

    # TF-IDF для вторых 5 предложений
    vectorizer2 = TfidfVectorizer()
    tfidf_part2 = vectorizer2.fit_transform(part2)
    headers_part2 = vectorizer2.get_feature_names_out()

    print(f"\n--- TF и IDF для вторых 5 предложений ({language}) ---")
    for i, sent in enumerate(part2):
        print(f"Предложение {i+1}: {sent}")
        tf_values = tfidf_part2[i].toarray()[0]
        for word, tf in zip(headers_part2, tf_values):
            print(f"  TF[{word}] = {tf:.4f}")
    for word, idf in zip(headers_part2, vectorizer2.idf_):
        print(f"IDF[{word}] = {idf:.4f}")

    df_tfidf_part2 = pd.DataFrame(tfidf_part2.toarray(), columns=headers_part2)
    df_tfidf_part2.to_excel(f"{language}_tfidf_part2.xlsx", index=False)

    # Создание векторов для двух частей как единого текста
    part1_combined = [" ".join(part1)]
    part2_combined = [" ".join(part2)]
    vectorizer_combined = TfidfVectorizer()
    tfidf_combined = vectorizer_combined.fit_transform(part1_combined + part2_combined)

    headers_combined = vectorizer_combined.get_feature_names_out()
    df_tfidf_combined = pd.DataFrame(tfidf_combined.toarray(), index=["Part1", "Part2"], columns=headers_combined)
    df_tfidf_combined.to_excel(f"{language}_tfidf_combined.xlsx", index=True)

    # Промежуточные TF и IDF таблицы
    df_tf = pd.DataFrame(vectorizer_combined.transform(part1_combined + part2_combined).toarray(), columns=headers_combined)
    df_tf.to_excel(f"{language}_tf_values.xlsx", index=False)

    df_idf = pd.DataFrame({'Word': headers_combined, 'IDF': vectorizer_combined.idf_})
    df_idf.to_excel(f"{language}_idf_values.xlsx", index=False)

    # Косинусное сходство между двумя частями текста
    similarity = cosine_similarity(tfidf_combined[0], tfidf_combined[1])[0][0]

    return similarity

# Чтение текстов из файлов
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

russian_text = read_file('russian.txt')
belarussian_text = read_file('belarussian.txt')
english_text = read_file('english.txt')

# Разбивка текста на предложения
textsRU = sent_tokenize(russian_text)[:10]  # Берём первые 10 предложений
textsBY = sent_tokenize(belarussian_text)[:10]
textsEN = sent_tokenize(english_text)[:10]

# Обработка русского текста
print("Обработка русского языка")
similarity_ru = process_language("ru", textsRU)
print(f"Косинусное сходство между частями текста (русский): {similarity_ru:.4f}")

# Обработка белорусского текста
print("Обработка белорусского языка")
similarity_by = process_language("be", textsBY)
print(f"Косинусное сходство между частями текста (белорусский): {similarity_by:.4f}")

# Обработка английского текста
print("Обработка английского языка")
similarity_en = process_language("en", textsEN)
print(f"Косинусное сходство между частями текста (английский): {similarity_en:.4f}")

print("Результаты сохранены в файлы.")
