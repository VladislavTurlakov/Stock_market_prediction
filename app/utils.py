import os
import requests
import pandas as pd
import logging
from io import StringIO
import time
from functools import lru_cache

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'moex_data')

def download_data(ticker, start_date=None, end_date=None, max_retries=3):
    """
    Загружает исторические данные по акции с сайта Московской биржи (MOEX) и сохраняет их в локальный CSV-файл.

    Input:
        ticker : str
            Биржевой тикер.
        start_date : str
            Начальная дата выборки в формате 'YYYY-MM-DD'.
        end_date : str
            Конечная дата выборки в формате 'YYYY-MM-DD'.
        max_retries : int
            Максимальное количество попыток загрузки (по умолчанию 3).

    Output:
        df : pd.DataFrame
            Таблица с историческими данными по тикеру.
    """
    base_url = f'https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.csv'
    all_data = []
    start = 0
    batch_size = 100  # Максимальное количество строк за один запрос

    for attempt in range(max_retries):
        try:
            while True:
                params = {
                    'start': start,
                    'limit': batch_size
                }
                if start_date:
                    params['from'] = start_date
                if end_date:
                    params['till'] = end_date

                response = requests.get(base_url, params=params)  # Запрос к API
                if response.status_code != 200:
                    raise Exception(f"HTTP ошибка {response.status_code}: {response.text}")

                # Читаем данные из ответа
                data_str = response.text
                lines = data_str.split('\n')

                # Находим начало и конец данных
                data_start = None
                data_end = None
                for i, line in enumerate(lines):
                    if line.startswith('BOARDID;TRADEDATE;'):
                        data_start = i
                    if line.startswith('history.cursor'):
                        data_end = i
                        break

                # Если данных нет — прерываем
                if data_start is None:
                    logging.info("Заголовок данных не найден. Прерываем цикл.")
                    break

                # Извлекаем данные
                data_lines = lines[data_start:data_end]
                if len(data_lines) <= 1:  # Только заголовок
                    logging.info("Строки данных не найдены. Прерываем цикл.")
                    break

                # Преобразуем в DataFrame
                data_str = '\n'.join(data_lines)
                df_batch = pd.read_csv(StringIO(data_str), delimiter=';', encoding='cp1251')

                # Логирование первых строк батча для диагностики
                logging.info(f"Фрагмент данных батча:\n{df_batch.head()}")

                all_data.append(df_batch)

                # Проверяем, есть ли еще данные
                if len(df_batch) < batch_size:
                    logging.info("Размер батча меньше лимита. Прерываем цикл.")
                    break

                start += batch_size
                time.sleep(0.5)  # Задержка между запросами

            # Если ничего не скачали
            if not all_data:
                raise Exception(f"Данные для {ticker} не найдены")

            # Объединяем все данные
            df = pd.concat(all_data, ignore_index=True)

            # Логирование объединённых данных для диагностики
            logging.info(f"Фрагмент объединённых данных:\n{df.head()}")

            # Сохраняем в файл
            os.makedirs(DATA_DIR, exist_ok=True)
            file_path = os.path.join(DATA_DIR, f'{ticker}.csv')
            df.to_csv(file_path, index=False, encoding='cp1251')
            logging.info(f"Данные загружены и сохранены в {file_path}. Общее количество строк: {len(df)}")
            return df

        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Все попытки завершились неудачей. Ошибка: {str(e)}")
                raise
            logging.warning(f"Попытка {attempt + 1} завершилась неудачей. Повторная попытка... Ошибка: {str(e)}")
            time.sleep(2 ** attempt)  # Экспоненциальная задержка

@lru_cache(maxsize=32)
def load_data(ticker):
    """
    Загружает и обрабатывает локальные данные по тикеру с Московской биржи (MOEX).

    Input:
        ticker : str
            Биржевой тикер.

    Output:
        df : pd.DataFrame
            DataFrame с обработанными данными, где:
                индекс : `TRADEDATE` (datetime),
                столбцы : котировки и метаданные (например, 'OPEN', 'CLOSE', 'VOLUME').
    """
    file_path = os.path.join(DATA_DIR, f'{ticker}.csv')

    try:
        # Читаем файл с учетом возможных ошибок кодировки
        with open(file_path, 'r', encoding='cp1251') as f:
            lines = f.readlines()

        # Находим начало и конец данных
        header_idx = None
        footer_idx = None
        for i, line in enumerate(lines):
            if line.startswith('BOARDID;TRADEDATE;'):
                header_idx = i
            if line.startswith('history.cursor'):
                footer_idx = i
                break

        if header_idx is None:
            raise Exception("Не удалось найти заголовок данных в CSV файле")

        # Читаем только данные
        if footer_idx is not None:
            data_lines = lines[header_idx:footer_idx]
        else:
            data_lines = lines[header_idx:]

        # Преобразуем в DataFrame
        data_str = '\n'.join(data_lines)
        df = pd.read_csv(StringIO(data_str), delimiter=';', encoding='cp1251')

        # Логирование первых строк загруженных данных для диагностики
        logging.info(f"Фрагмент загруженных данных:\n{df.head()}")

        # Очистка данных
        df = df.dropna(subset=['TRADEDATE'])
        df = df[df['TRADEDATE'].str.contains(r'\d{4}-\d{2}-\d{2}', na=False)]

        # Преобразуем дату и устанавливаем как индекс
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format='%Y-%m-%d')
        df.set_index('TRADEDATE', inplace=True)
        df.sort_index(inplace=True)  # Сортируем по дате

        # Преобразуем числовые колонки в float
        numeric_cols = ['NUMTRADES', 'VALUE', 'OPEN', 'LOW', 'HIGH', 'LEGALCLOSEPRICE',
                        'WAPRICE', 'CLOSE', 'VOLUME', 'MARKETPRICE2', 'MARKETPRICE3',
                        'ADMITTEDQUOTE', 'MP2VALTRD', 'MARKETPRICE3TRADESVALUE',
                        'ADMITTEDVALUE', 'WAVAL', 'TRENDCLSPR']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Проверка наличия пропущенных значений в столбце CLOSE
        if df['CLOSE'].isnull().any():
            logging.warning("В столбце 'CLOSE' есть пропущенные значения.")
            logging.info(f"Количество пропущенных значений в 'CLOSE': {df['CLOSE'].isnull().sum()}")

        # Проверяем необходимые колонки
        required_columns = {'CLOSE'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise Exception(f"Отсутствуют необходимые столбцы: {missing}")

        # Проверка на пустой DataFrame
        if df.empty:
            raise Exception(f"Тикер {ticker} не распознан: данные отсутствуют.")

        logging.info(f"DataFrame загружен. Размер: {df.shape}, Диапазон дат: {df.index.min()} до {df.index.max()}")
        return df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных для {ticker}: {str(e)}")
        raise
