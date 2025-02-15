import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import dash
from dash import dcc, html
import plotly.graph_objs as go
import os

# Создание модели
def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # 2 выхода: координаты X и Y
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

# Генерация данных
def generate_data():
    time = np.linspace(0, 365, 1000)
    mars_x = np.sin(time / 365 * 2 * np.pi)  
    mars_y = np.cos(time / 365 * 2 * np.pi)  

    X_train = time[:-1].reshape(-1, 1)  
    y_train = np.column_stack((mars_x[1:], mars_y[1:])) 

    future_time = np.linspace(365, 730, 1000).reshape(-1, 1)

    return X_train, y_train, mars_x, mars_y, future_time

# Обучение модели
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=1000, batch_size=32)

# Прогнозирование
def predict(model, future_time):
    return model.predict(future_time)

# Создание графика
def create_graph():
    X_train, y_train, mars_x, mars_y, future_time = generate_data()
    model = create_model()
    train_model(model, X_train, y_train)
    predictions = predict(model, future_time)

    figure = {
        'data': [
            go.Scatter(x=mars_x, y=mars_y, mode='lines', name='Исторические данные'),
            go.Scatter(x=predictions[:, 0], y=predictions[:, 1], mode='lines', name='Предсказания нейросети', line=dict(dash='dash'))
        ],
        'layout': go.Layout(
            title="Предсказание движения Марса",
            xaxis={'title': 'Координаты X'},
            yaxis={'title': 'Координаты Y'}
        )
    }
    return figure

# Создание приложения Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Предсказание движения Марса", style={"text-align": "center"}),
    dcc.Graph(
        id='mars-orbit-graph',
        figure=create_graph()  # Вставляем график
    )
])

# Запуск приложения
if __name__ == '__main__':
    # Получаем порт из переменной окружения PORT для деплоя
    port = int(os.environ.get("PORT", 8050))  # 8050 - порт по умолчанию
    app.run_server(debug=True, host='0.0.0.0', port=port)  # Указываем слушать на всех интерфейсах
    gunicorn app:mars_app.server
