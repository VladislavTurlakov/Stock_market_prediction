import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .utils import DATA_DIR
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

class LSTMModel(nn.Module):
    """
    Модель нейронной сети на основе LSTM для обработки последовательностей.

    Input:
        input_size : int
            Размер входных признаков.
        hidden_size : int
            Размер скрытого состояния.
        num_layers : int
            Количество рекуррентных слоев.
        output_size : int
            Размер выходного вектора.

    Output:
        out : torch.Tensor
            Выход модели размерности, содержащий предсказания для последнего временного шага.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_data(ticker):
    """
    Загружает и подготавливает исторические данные по тикеру из CSV-файла.

    Функция читает CSV с котировками, очищает строки с некорректными датами,
    преобразует даты в индекс, приводит числовые колонки к float, заполняет
    пропуски в 'CLOSE' линейной интерполяцией и проверяет наличие необходимых столбцов.

    Input:
        ticker : str
            Тикер финансового инструмента.

    Output:
        pandas.DataFrame
            Подготовленный DataFrame:
            - Индекс: столбец `TRADEDATE` (datetime, отсортированный).
            - Числовые колонки приведены к float.
            - Пропуски в `CLOSE` заполнены интерполяцией.
    """

    # Формируем путь к CSV-файлу по тикеру
    file_path = os.path.join(DATA_DIR, f'{ticker}.csv')

    try:
        # Читаем файл с помощью pandas, указывая кодировку и разделитель
        df = pd.read_csv(file_path, encoding='cp1251', delimiter=',')
        logging.info(f"Данные прочитаны из файла. Размер: {df.shape}")

        # Логирование первых строк загруженных данных для диагностики
        logging.info(f"Загруженные данные:\n{df.head()}")

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

        # Заполняем пропущенные значения в CLOSE (линейная интерполяция)
        if df['CLOSE'].isnull().any():
            logging.warning("В столбце 'CLOSE' есть пропущенные значения.")
            missing_count = df['CLOSE'].isnull().sum()
            logging.info(f"Количество пропущенных значений в 'CLOSE': {missing_count}")
            df['CLOSE'] = df['CLOSE'].interpolate(method='time')
            logging.info(f"Заполнено {missing_count} пропущенных значений в 'CLOSE'")

        # Проверяем необходимые колонки
        required_columns = {'CLOSE'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise Exception(f"Отсутствуют необходимые столбцы: {missing}")

        # Финальный лог: размер таблицы и диапазон дат
        logging.info(
            f"DataFrame загружен. Размер: {df.shape}, "
            f"Диапазон дат: {df.index.min()} до {df.index.max()}"
        )
        return df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных для {ticker}: {str(e)}")
        raise Exception(f"Ошибка при загрузке данных для {ticker}: {str(e)}")

def create_dataset(data, time_step=1):
    """
    Формирует обучающий датасет для временного ряда.

    Input:
        data : numpy.ndarray
            Массив данных.
        time_step : int, optional
            Количество предыдущих шагов, используемых для предсказания следующего значения.

    Output:
        tuple of numpy.ndarray
            X — массив входных последовательностей формы (n_samples, time_step).
            Y — массив целевых значений формы (n_samples,).
    """
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    logging.info(f"Размер датасета после создания: (X: {len(X)}, Y: {len(Y)})")
    return np.array(X), np.array(Y)

def train_model(ticker, save_dir='models'):
    """
    Обучает LSTM-модель прогнозирования цен по тикеру и сохраняет результаты.

    Функция загружает данные по тикеру, выполняет нормализацию и формирование выборки,
    обучает модель LSTM на временных рядах, сохраняет обученную модель, скейлер и график потерь.

    Input:
        ticker : str
            Тикер финансового инструмента.
        save_dir : str, optional
            Папка для сохранения модели, скейлера и графика потерь.

    Output:
        model : torch.nn.Module
            Обученная LSTM-модель.
        scaler : sklearn.preprocessing.MinMaxScaler
            Скалер для обратного преобразования предсказаний.
    """
    df = load_data(ticker)
    df = df[['CLOSE']]

    # Убедимся, что нет пропущенных значений
    df = df.dropna()

    # Проверим, достаточно ли данных
    if len(df) < 100:
        raise ValueError(f"Недостаточно данных для обучения. Необходимо минимум 100 дней, получено {len(df)}")

    dataset = df.values
    dataset = dataset.astype('float32')

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    time_step = 60
    X, Y = create_dataset(dataset, time_step)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, Y_test.unsqueeze(1))

        # Сохраняем потери
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        logging.info(
            f"Эпоха [{epoch + 1}/{num_epochs}], Тренировочная потеря: {loss.item():.4f}, Потеря валидации: {val_loss.item():.4f}")

    # Сохранение модели и скалера
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f'{ticker}_model.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Модель сохранена в {model_save_path}")

    scaler_save_path = os.path.join(save_dir, f'{ticker}_scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    logging.info(f"Скалер сохранен в {scaler_save_path}")

    # Графики потерь
    sns.set(style='whitegrid')

    plt.figure(figsize=(12, 7))

    plt.plot(range(1, num_epochs + 1), train_losses, label='Тренировочная ошибка', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, label='Ошибка на валидации', color='red', linewidth=2, linestyle='--')

    plt.xlabel('Эпоха', fontsize=14)
    plt.ylabel('MSE Ошибка', fontsize=14)
    plt.title(f'График изменения потерь (MSE) на обучении и валидации\nдля тикера {ticker}', fontsize=16, weight='bold')

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Сохраняем график
    loss_plot_path = os.path.join(save_dir, f'{ticker}_loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()

    logging.info(f"График потерь сохранён в {loss_plot_path}")

    return model, scaler

def predict_stock_price(model, scaler, data):
    """
    Делает предсказание цены акции с помощью обученной LSTM-модели.

    Функция нормализует данные, использует последние 60 наблюдений,
    подаёт их в модель и возвращает восстановленное предсказание в исходном масштабе.

    Input:
        model : torch.nn.Module
            Обученная LSTM-модель.
        scaler : sklearn.preprocessing.MinMaxScaler
            Скалер, использованный при обучении для нормализации данных.
        data : numpy.ndarray
            Исходные данные временного ряда (двумерный массив с ценами).

    Output:
        predicted_price[0][0] : float
            Предсказанная цена акции в исходном масштабе.
    """
    model.eval()
    with torch.no_grad():
        # Нормализация данных
        data = scaler.transform(data)
        data = data.reshape(-1, 1)
        data = data[-60:]  # Используем последние 60 дней
        data = data.reshape(1, 60, 1)
        data = torch.tensor(data, dtype=torch.float32)
        predicted_price = model(data)
        predicted_price = scaler.inverse_transform(predicted_price.numpy())
    return predicted_price[0][0]

def predict_future_prices(model, scaler, initial_data, steps=30):
    """
    Предсказывает будущие цены на несколько дней вперёд с помощью обученной LSTM-модели.

    На вход подаются последние 60 дней данных, модель поочередно предсказывает
    следующие значения, и каждое новое предсказание добавляется в последовательность
    для дальнейшего прогноза.

    Input:
        model : torch.nn.Module
            Обученная LSTM-модель.
        scaler : sklearn.preprocessing.MinMaxScaler
            Скалер, использованный при обучении для нормализации данных.
        initial_data : numpy.ndarray
            Исходные данные (двумерный массив, содержащий последние 60 дней цен).
        steps : int, optional
            Количество дней для предсказания (по умолчанию 30).

    Output:
        future_predictions: list[float]
            Список предсказанных цен на заданное количество шагов вперёд.
    """
    model.eval()
    future_predictions = []

    # Копируем последние 60 дней данных
    current_sequence = initial_data.copy()

    for _ in range(steps):
        with torch.no_grad():
            # Нормализуем текущую последовательность
            normalized_seq = scaler.transform(current_sequence)
            normalized_seq = normalized_seq.reshape(1, -1, 1)

            # Преобразуем в тензор
            tensor_seq = torch.tensor(normalized_seq, dtype=torch.float32)

            # Получаем предсказание
            prediction = model(tensor_seq)

            # Денормализуем предсказание
            prediction_price = scaler.inverse_transform(prediction.numpy())[0][0]
            future_predictions.append(prediction_price)

            # Обновляем последовательность, добавляя предсказание и удаляя самый старый элемент
            current_sequence = np.append(current_sequence[1:], [[prediction_price]], axis=0)

    return future_predictions
