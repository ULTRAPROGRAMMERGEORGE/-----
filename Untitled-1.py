# %%
import pandas as pd
import numpy as np
# Загружаем файл
file_path = "dt.csv"
df = pd.read_csv(file_path)
df.head()

# %%
import pandas as pd
import numpy as np
import re

# Курсы валют
exchange_rates = {
    "$": 92.5,   # 1 доллар = 92.5 руб.
    "€": 100.2,  # 1 евро = 100.2 руб.
    "₽": 1       # Рубли остаются без изменений
}

# Функция для обработки зарплаты (извлечение чисел и перевод валют)
def parse_salary(salary):
    if pd.isna(salary) or "Не указана" in str(salary):
        return np.nan
    
    salary = salary.replace(" ", "")  # Убираем пробелы
    match = re.match(r"([\$\€₽]?)(\d+)-?(\d+)?", salary)  # Ищем валюту и числа
    if not match:
        return np.nan

    currency, min_val, max_val = match.groups()
    min_val = int(min_val)
    max_val = int(max_val) if max_val else min_val  # Если одно число, берем его же
    avg_salary = (min_val + max_val) / 2  # Среднее значение

    return avg_salary * exchange_rates.get(currency, 1)  # Конвертируем в рубли

# Функция для форматирования зарплаты (100000 → 100K)
def format_salary(value):
    if pd.isna(value):
        return np.nan
    if value >= 1000:
        return int(value // 1000)
    return str(value)

# Функция для обработки опыта работы
def parse_experience(exp):
    if pd.isna(exp):
        return np.nan
    exp = str(exp)
    if "Более 6 лет" in exp:
        return 7
    if "От 3 до 6 лет" in exp:
        return 4.5
    if "От 1 года до 3 лет" in exp:
        return 2
    if "Нет опыта" in exp:
        return 0
    return np.nan

# Загружаем файл
file_path = "dt.csv"
df = pd.read_csv(file_path)

# Обрабатываем зарплату
df["Зарплата"] = df["Зарплата"].apply(parse_salary)
df["Зарплата"] = df["Зарплата"].apply(format_salary)

# Обрабатываем опыт работы
df["Опыт работы"] = df["Опыт работы"].apply(parse_experience)

# Выводим результат
print(df[["Зарплата", "Опыт работы"]].head())

# Выведем первые строки после обработки

# %%
df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Убираем строки с пропущенными значениями
df_ed = df[df['Требования'].str.contains('Высшее образование', na=False)]
df_wed = df[~df['Требования'].str.contains('Высшее образование', na=False)]

df_ed.head()


# %%
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
df_with_higher_edu = df_ed
df_without_higher_edu = df_wed
# Предположим, что df_with_higher_edu и df_without_higher_edu уже созданы
n = range(len(df_with_higher_edu))  # Индексы строк
n2 = range(len(df_without_higher_edu))  

plt.scatter(n, df_with_higher_edu["Зарплата"], label="С высшим образованием", color="green")
plt.scatter(n2, df_without_higher_edu["Зарплата"], label="Без высшего образования", color="orange")

# Добавление подписей и легенды
plt.xlabel("Группы")
plt.ylabel("Зарплата в тысячах")
plt.legend()
plt.title("Сравнение зарплат")

plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Проверим тип данных в зарплате
print(df_with_higher_edu.dtypes)
print(df_without_higher_edu.dtypes)

# Преобразуем зарплату в числовой формат, если нужно
df_with_higher_edu["Зарплата"] = pd.to_numeric(df_with_higher_edu["Зарплата"], errors="coerce")
df_without_higher_edu["Зарплата"] = pd.to_numeric(df_without_higher_edu["Зарплата"], errors="coerce")

# Заполним пропущенные значения средним значением
df_with_higher_edu["Зарплата"].fillna(df_with_higher_edu["Зарплата"].mean(), inplace=True)
df_without_higher_edu["Зарплата"].fillna(df_without_higher_edu["Зарплата"].mean(), inplace=True)

# Добавляем метки классов
df_with_higher_edu["Высшее_образование"] = 1
df_without_higher_edu["Высшее_образование"] = 0

# Объединяем данные
df2 = pd.concat([df_with_higher_edu, df_without_higher_edu])

# Проверяем, как выглядят данные
print(df2.head())

# Выбираем признаки (зарплату) и целевую переменную
X = df2[["Зарплата"]]
y = df2["Высшее_образование"]

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Делаем предсказания
y_pred = clf.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Визуализация данных
plt.scatter(df2["Зарплата"], df2["Высшее_образование"], c=df2["Высшее_образование"], cmap="coolwarm", alpha=0.6)
plt.xlabel("Зарплата")
plt.ylabel("Высшее образование (1 - да, 0 - нет)")
plt.title("Зависимость высшего образования от зарплаты")
plt.colorbar(label="Класс (0 - без высшего, 1 - с высшим)")
plt.show()



