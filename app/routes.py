from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .models import LSTMModel, load_data, train_model, predict_stock_price, predict_future_prices
from .utils import download_data
import torch
import numpy as np
import pandas as pd
import os
import logging
from datetime import timedelta
import joblib

logging.basicConfig(level=logging.INFO)

router = APIRouter()

class PredictionRequest(BaseModel):
    """
    Модель валидации входных данных для API-запроса предсказания.

    Input:
        ticker : str
            Тикер акции.
        date : str
            Дата, на которую запрашивается прогноз (в формате 'YYYY-MM-DD').
    """
    ticker: str
    date: str

def prepare_dataset(df):
    """
    Подготавливает датасет для обучения или предсказания на основе входного DataFrame.

    Input:
        df : pandas.DataFrame
            Таблица с историческими данными, должна содержать колонку 'CLOSE'.

    Output:
        tuple:
            df : pandas.DataFrame
                Очищенный DataFrame только с колонкой 'CLOSE'.
            dataset : numpy.ndarray
                Массив цен закрытия типа float32.
    """
    df = df[['CLOSE']].dropna()
    dataset = df.values.astype('float32')
    return df, dataset

def load_or_train_model(ticker):
    """
    Загружает сохранённую модель и нормализатор для указанного тикера или обучает новую,
    если сохранённых файлов не найдено.

    Input:
        ticker : str
            Биржевой тикер компании или инструмента (например, 'SBER').

    Output:
        tuple :
            model : LSTMModel
                Загруженная или вновь обученная модель.
            scaler : sklearn.preprocessing.MinMaxScaler
                Нормализатор для приведения данных к диапазону, использованному при обучении.
    """
    model_save_path = os.path.join('models', f'{ticker}_model.pth')
    scaler_save_path = os.path.join('models', f'{ticker}_scaler.pkl')

    if not os.path.exists(model_save_path) or not os.path.exists(scaler_save_path):
        logging.info("Модель не найдена. Обучение новой модели...")
        model, scaler = train_model(ticker)
        joblib.dump(scaler, scaler_save_path)
        torch.save(model.state_dict(), model_save_path)
    else:
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        scaler = joblib.load(scaler_save_path)

    return model, scaler

def predict_historical_price(model, scaler, dataset, date, df):
    """
    Предсказывает цену акции на указанную историческую дату, используя данные за предыдущие 60 дней.

    Input:
        model : torch.nn.Module
            Обученная LSTM-модель.
        scaler : sklearn.preprocessing.MinMaxScaler
            Нормализатор данных.
        dataset : np.ndarray
            Массив цен закрытия в формате float32.
        date : str
            Дата в формате 'YYYY-MM-DD', для которой нужно сделать прогноз.
        df : pandas.DataFrame
            Исходные данные с индексом по датам.

    Output:
        dict :
            predicted_price : float
                Предсказанная цена на указанную дату.
            is_future : bool
                Признак, указывающий, что прогноз относится к прошлым данным.
    """
    # Находим индекс нужной даты в DataFrame
    date_index = df.index.get_loc(date)

    # Проверка: достаточно ли данных (минимум 60 дней до указанной даты)
    if date_index < 60:
        raise Exception(f"Недостаточно данных до {date}. Необходимо 60 дней перед датой.")

    # Берём 60 предыдущих дней для прогноза
    last_60_days = dataset[date_index - 60:date_index]
    predicted_price = predict_stock_price(model, scaler, last_60_days)
    logging.info(f"Предсказанная цена для {date}: {predicted_price}")

    return {
        "predicted_price": float(predicted_price),
        "is_future": False
    }

def predict_future_price(model, scaler, dataset, date, df):
    """
    Предсказывает цену акции на будущую дату, используя последние 60 дней исторических данных.

    Input:
        model : torch.nn.Module
            Обученная LSTM-модель.
        scaler : sklearn.preprocessing.MinMaxScaler
            Нормализатор данных.
        dataset : np.ndarray
            Массив цен закрытия в формате float32.
        date : datetime.date или datetime.datetime
            Будущая дата, для которой требуется прогноз.
        df : pandas.DataFrame
            Исходные данные с индексом по датам.

    Output:
        dict:
            predicted_price : float
                Предсказанная цена на указанную дату.
            is_future : bool
                Всегда True, так как прогноз на будущее.
            days_ahead : int
                Количество дней вперёд от последней доступной даты.
    """
    # Вычисляем, сколько дней прошло с последней доступной даты до целевой
    days_ahead = (date - df.index[-1]).days
    if days_ahead <= 0:
        raise Exception(f"Дата {date} находится раньше последней доступной даты {df.index[-1].strftime('%Y-%m-%d')}")

    # Берём последние 60 дней из датасета
    last_60_days = dataset[-60:]

    # Генерируем прогноз на заданное количество шагов вперёд
    future_prices = predict_future_prices(model, scaler, last_60_days, steps=days_ahead)

    # Берём цену именно для нужной даты (последний шаг)
    predicted_price = future_prices[-1]
    logging.info(f"Предсказанная будущая цена для {date}: {predicted_price}")

    return {
        "predicted_price": float(predicted_price),
        "is_future": True,
        "days_ahead": days_ahead
    }

@router.post("/predict/")
async def predict_price(request: PredictionRequest):
    """
    Обрабатывает POST-запрос на предсказание цены акции по тикеру и дате.

    Input:
        request (PredictionRequest): Объект запроса, содержащий:
            ticker : str
                Биржевой тикер.
            date : str
                Дата для прогноза в формате "YYYY-MM-DD".

    Output:
        dict:
            predicted_price : float
                Предсказанная цена на указанную дату.
            is_future : bool
                Признак, указывающий, что прогноз относится к прошлым или будущим данным.
            days_ahead : int
                Количество дней вперёд от последней доступной даты.
    """
    # Извлекаем тикер и дату из тела запроса
    ticker = request.ticker
    date_str = request.date

    logging.info(f"Получен запрос: ticker={ticker}, date={date_str}")

    try:
        # Конвертируем дату из строки в datetime
        date = pd.to_datetime(date_str)
        start_date = '2014-01-01'
        end_date = (date + timedelta(days=365)).strftime('%Y-%m-%d')

        # Загружаем данные с биржи
        download_data(ticker, start_date=start_date, end_date=end_date)
        logging.info(f"Данные загружены для тикера: {ticker}")

        # Загружаем и обрабатываем CSV
        df = load_data(ticker)
        logging.info(f"Данные загружены для тикера: {ticker}. Диапазон дат: {df.index.min()} до {df.index.max()}")

        # Заполняем пропущенные даты
        date_range = pd.date_range(start=df.index.min(), end=max(df.index.max(), date), freq='D')
        df = df.reindex(date_range)
        df['CLOSE'] = df['CLOSE'].interpolate(method='time')

        df, dataset = prepare_dataset(df)

        # Проверяем, достаточно ли данных (минимум 60 дней для LSTM)
        if len(dataset) < 60:
            raise Exception(f"Недостаточно данных для прогнозирования. Необходимо 60 дней, имеется {len(dataset)}")

        # Загружаем готовую модель или обучаем новую
        model, scaler = load_or_train_model(ticker)

        # Теперь все даты заполнены, можно предсказывать
        if date <= df.index[-1]:
            return predict_historical_price(model, scaler, dataset, date, df)
        else:
            return predict_future_price(model, scaler, dataset, date, df)

    except Exception as e:
        error_message = str(e)
        if "Не удалось найти заголовок данных в CSV файле" in error_message:
            error_message = "Тикер не распознан: данные отсутствуют."
        elif "Тикер" in error_message and "не распознан" in error_message:
            error_message = "Тикер не распознан: данные отсутствуют."
        elif "Neither `start` nor `end` can be NaT" in error_message:
            error_message = "Тикер не распознан: данные отсутствуют."
        logging.error(f"Ошибка при обработке запроса: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

@router.get("/get_min_date/")
async def get_min_date(ticker: str):
    """
    Возвращает минимальную и максимальную доступные даты для прогноза по указанному тикеру.

    Input:
        ticker : str
            Биржевой тикер.

    Output:
        dict :
            min_date : str
                Минимальная дата прогноза.
            max_date : str
                Максимальная дата прогноза.
    """
    try:
        # Загружаем данные по тикеру
        df = load_data(ticker)
        # Подготавливаем датасет (оставляем только CLOSE)
        df, _ = prepare_dataset(df)

        # Проверка: достаточно ли данных для прогноза
        if len(df) < 60:
            raise Exception(f"Недостаточно данных (необходимо 60 дней, имеется {len(df)})")

        min_date = df.index[59]  # Первая дата, для которой есть 60 предыдущих дней
        max_date = df.index[-1]  # Последняя доступная дата в данных

        logging.info(f"Диапазон дат для {ticker}: {min_date} до {max_date}")
        return {
            "min_date": min_date.strftime('%Y-%m-%d'),
            "max_date": max_date.strftime('%Y-%m-%d')
        }

    except Exception as e:
        error_message = str(e)
        if "Не удалось найти заголовок данных в CSV файле" in error_message:
            error_message = "Тикер не распознан: данные отсутствуют."
        elif "Тикер" in error_message and "не распознан" in error_message:
            error_message = "Тикер не распознан: данные отсутствуют."
        logging.error(f"Ошибка при получении диапазона дат: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
