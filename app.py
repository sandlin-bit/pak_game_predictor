from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

def load_data():
    if os.path.exists("pak_lottery_data.csv"):
        return pd.read_csv("pak_lottery_data.csv")
    return pd.DataFrame(columns=['number'])

def save_number(num):
    df = load_data()
    df = df.append({'number': int(num)}, ignore_index=True)
    df.to_csv("pak_lottery_data.csv", index=False)

def prepare_data(df):
    numbers = df['number'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(numbers)
    gen = TimeseriesGenerator(scaled, scaled, length=5, batch_size=1)
    X = np.array([x[0] for x in gen])
    y = np.array([1 if x[1][0] >= 0.5 else 0 for x in gen])
    return X, y, scaler, scaled

def train_model(X, y):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, verbose=0)
    return model

def predict_next(model, scaler, scaled):
    last_sequence = scaled[-5:].reshape(1, 5, 1)
    pred = model.predict(last_sequence)[0][0]
    return "Big" if pred >= 0.5 else "Small"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "number" in request.form:
            number = request.form["number"]
            if number.isdigit() and 0 <= int(number) <= 9:
                save_number(number)
            else:
                prediction = "Please enter a valid number (0–9)."
        if "predict" in request.form:
            df = load_data()
            if len(df) >= 6:
                X, y, scaler, scaled = prepare_data(df)
                model = train_model(X, y)
                prediction = predict_next(model, scaler, scaled)
            else:
                prediction = "Add more data first (at least 6 numbers)."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
