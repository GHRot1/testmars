import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import dash
from dash import dcc, html
import plotly.graph_objs as go
import os
import logging
from dash.dependencies import Input, Output, State
import flask
import threading
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация Flask и Dash
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Глобальная переменная для хранения обученной модели
global_model = None
model_ready = False
training_thread = None

# Создание модели
def create_model():
    try:
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(2)  # 2 выхода: координаты X и Y
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Ошибка при создании модели: {str(e)}")
        raise

# Генерация данных
def generate_data():
    try:
        time = np.linspace(0, 365, 1000)
        mars_x = np.sin(time / 365 * 2 * np.pi)  
        mars_y = np.cos(time / 365 * 2 * np.pi)  
        X_train = time[:-1].reshape(-1, 1)  
        y_train = np.column_stack((mars_x[1:], mars_y[1:])) 
        future_time = np.linspace(365, 730, 1000).reshape(-1, 1)
        return X_train, y_train, mars_x, mars_y, future_time
    except Exception as e:
        logger.error(f"Ошибка при генерации данных: {str(e)}")
        raise

# Обучение модели
def train_model(model, X_train, y_train):
    try:
        logger.info("Начинаю обучение модели...")
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        logger.info("Обучение завершено.")
        return model
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

# Прогнозирование
def predict(model, future_time):
    try:
        logger.info("Запуск прогнозирования...")
        return model.predict(future_time)
    except Exception as e:
        logger.error(f"Ошибка при прогнозировании: {str(e)}")
        raise

# Функция для асинхронной инициализации модели
def initialize_model_async():
    global global_model, model_ready
    try:
        X_train, y_train, _, _, _ = generate_data()
        model = create_model()
        global_model = train_model(model, X_train, y_train)
        model_ready = True
        logger.info("Модель успешно инициализирована и обучена.")
    except Exception as e:
        logger.error(f"Ошибка при асинхронной инициализации модели: {str(e)}")

# Начальная инициализация
@app.callback(
    Output("model-status", "children"),
    Input("model-interval", "n_intervals")
)
def update_model_status(n):
    if model_ready:
        return "Модель готова к использованию"
    else:
        return "Модель обучается... Пожалуйста, подождите."

# Функция для создания графиков
def create_initial_graph():
    try:
        _, _, mars_x, mars_y, _ = generate_data()
        # Создаем начальный график только с историческими данными
        figure = {
            'data': [
                go.Scatter(x=mars_x, y=mars_y, mode='lines', name='Исторические данные')
            ],
            'layout': go.Layout(
                title="Предсказание движения Марса",
                xaxis={'title': 'Координаты X'},
                yaxis={'title': 'Координаты Y'}
            )
        }
        return figure
    except Exception as e:
        logger.error(f"Ошибка при создании начального графика: {str(e)}")
        # Возвращаем пустой график в случае ошибки
        return {
            'data': [],
            'layout': go.Layout(
                title="Ошибка при загрузке данных",
                xaxis={'title': 'Координаты X'},
                yaxis={'title': 'Координаты Y'}
            )
        }

# Обновление графика с предсказаниями
@app.callback(
    Output('mars-orbit-graph', 'figure'),
    Input('update-button', 'n_clicks'),
    Input('model-interval', 'n_intervals')
)
def update_graph(n_clicks, n_intervals):
    global global_model, model_ready
    
    if not model_ready:
        return create_initial_graph()
    
    try:
        _, _, mars_x, mars_y, future_time = generate_data()
        predictions = predict(global_model, future_time)
        
        figure = {
            'data': [
                go.Scatter(x=mars_x, y=mars_y, mode='lines', name='Исторические данные'),
                go.Scatter(x=predictions[:, 0], y=predictions[:, 1], mode='lines', 
                           name='Предсказания нейросети', line=dict(dash='dash'))
            ],
            'layout': go.Layout(
                title="Предсказание движения Марса",
                xaxis={'title': 'Координаты X'},
                yaxis={'title': 'Координаты Y'}
            )
        }
        return figure
    except Exception as e:
        logger.error(f"Ошибка при обновлении графика: {str(e)}")
        return create_initial_graph()

# Создание макета приложения
app.layout = html.Div([
    html.H1("Предсказание движения Марса", style={"text-align": "center"}),
    html.Div(id="model-status", style={"text-align": "center", "margin": "20px"}),
    dcc.Graph(
        id='mars-orbit-graph',
        figure=create_initial_graph()
    ),
    html.Button('Обновить график', id='update-button', style={"margin": "20px"}),
    dcc.Interval(
        id='model-interval',
        interval=1000,  # обновление каждую секунду
        n_intervals=0
    )
])

# Запуск сервера
if __name__ == '__main__':
    # Запуск асинхронной инициализации модели
    training_thread = threading.Thread(target=initialize_model_async)
    training_thread.daemon = True
    training_thread.start()
    
    # Получаем порт из переменной окружения PORT
    port = int(os.environ.get("PORT", 8050))
    
    # Режим debug отключаем для production
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Запуск сервера
    app.run_server(debug=debug_mode, host='0.0.0.0', port=port)
else:
    # Для запуска через WSGI сервер (gunicorn)
    # Запуск асинхронной инициализации модели
    training_thread = threading.Thread(target=initialize_model_async)
    training_thread.daemon = True
    training_thread.start()
