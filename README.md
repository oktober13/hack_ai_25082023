# Приложение сопоставления адресов
![image](https://github.com/oktober13/hack_ai_25082023/assets/114009321/81f47873-bffd-4fec-b83a-9ef7bcd0c895)

Приложение сопоставления адресов представляет собой скрипт на языке Python, 
который использует алгоритм ближайших соседей для сопоставления запросов адресов 
с наиболее похожими адресами из заданного набора данных. 
Также он вычисляет взвешенный показатель схожести на основе различных частей адресов.

## Подготовка данных

Для обработки тренировочных данных и запросов, вам потребуется использовать два дополнительных модуля:

1. `preproc_train.py` - модуль, который выполняет предобработку тренировочных данных. Он предоставляет функцию `merge_data` для объединения данных из разных источников.

2. `preproc_query.py` - модуль, который выполняет предобработку данных запросов. Он содержит функцию `replacer` для замены аббревиаций и форматирования адресов.

## Использование модулей предобработки

### Модуль `preproc_train.py`

Этот модуль предназначен для обработки тренировочных данных перед их использованием в Матчере Адресов. Он выполняет следующие шаги:

- Чтение данных из CSV файлов.
- Предобработка данных путем объединения и фильтрации.

Пример использования:

```
from preproc_train import merge_data

# Путь к файлам данных
area_file = 'area.csv'
areatype_file = 'areatype.csv'
# ... другие пути к файлам ...

# Выполняем предобработку
merged_data = merge_data(area_file, areatype_file, building_file, district_file, geonim_file, subrf_file, town_file, prefix_file)
```

### Модуль `preproc_query.py`

Этот модуль предназначен для предобработки данных запросов перед их использованием в Матчере Адресов. Он выполняет замену аббревиаций и форматирование адресов.

Пример использования:
```
from preproc_query import replacer

# Загружаем данные запросов
query_data = pd.read_csv('query_data.csv')

# Выполняем предобработку
query_data_processed = replacer(query_data)

```

## Требования

Убедитесь, что у вас установлены следующие зависимости:
```
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.25.2
```

Вы можете установить эти зависимости с помощью следующей команды:
```
pip install pandas scikit-learn numpy
```

## Использование

Скачайте или склонируйте этот репозиторий на ваш компьютер.

Перейдите в директорию проекта в вашем терминале.

Создайте виртуальное окружение (опционально, но рекомендуется):

```
python3 -m venv venv
source venv/bin/activate
```

Разместите ваши данные для обучения (CSV-файл с информацией об адресах) в директории проекта с именем файла result.csv.

Измените словарь weights в скрипте main.py, чтобы настроить веса для различных частей адреса.

Запустите скрипт для сопоставления адресов и создания CSV-файла с результатами:

```
python main.py
```

Для одиночного запроса адреса вы можете использовать метод find_matching_address для создания JSON-ответа:

```
from main import AddressMatcher

address_matcher = AddressMatcher(weights)
address_matcher.load_train_data('result.csv')

query_address = "аптерский 18 спб"
json_response = address_matcher.find_matching_address(query_address)
print(json_response)
```

## Конфигурация

Вы можете настроить веса для различных частей адреса в словаре weights в скрипте v3.py.


## Лицензия

Этот проект распространяется без лицензиии.

*Не стесняйтесь вносить свой вклад в этот проект, создавая issue или отправляя pull request'ы!*
