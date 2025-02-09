import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import io
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

class MarsOrbitModel:
    def __init__(self):
        self.model = self.create_model()
        self.train_model()
    
    def create_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(1,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(2)  # 2 выхода: координаты X и Y
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def generate_data(self):
        time = np.linspace(0, 365, 1000)
        mars_x = np.sin(time / 365 * 2 * np.pi)  
        mars_y = np.cos(time / 365 * 2 * np.pi)  

        X_train = time[:-1].reshape(-1, 1)  
        y_train = np.column_stack((mars_x[1:], mars_y[1:])) 

        future_time = np.linspace(365, 730, 1000).reshape(-1, 1)

        return X_train, y_train, mars_x, mars_y, future_time

    def train_model(self):
        X_train, y_train, _, _, _ = self.generate_data()
        self.model.fit(X_train, y_train, epochs=1000, batch_size=32)

    def predict(self, future_time):
        return self.model.predict(future_time)

class MarsOrbitApp:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.model = MarsOrbitModel()
        self.setup_layout()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Предсказание движения Марса", style={"text-align": "center"}),
            html.Div([
                dcc.Graph(
                    id='mars-orbit-graph',
                    figure=self.create_graph()
                )
            ])
        ])

    def create_graph(self):
        _, _, mars_x, mars_y, future_time = self.model.generate_data()
        predictions = self.model.predict(future_time)

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

    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    mars_app = MarsOrbitApp()  # Создание экземпляра приложения
    mars_app.run()  # Запуск приложения
