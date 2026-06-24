"""CNN-LSTM time-series demo — synthetic drilling sensors, 2-minute run."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise SystemExit("Install: pip install tensorflow scikit-learn numpy")


def synthetic_well_log(n: int = 800) -> np.ndarray:
    depth = np.linspace(0, 1, n)
    dtco = 40 + 10 * np.sin(depth * 12) + np.random.normal(0, 1, n)
    rop = 20 + 5 * np.cos(depth * 8) + np.random.normal(0, 0.8, n)
    gr = 60 + 15 * np.sin(depth * 5) + np.random.normal(0, 1.2, n)
    return np.column_stack([dtco, rop, gr])


def windows(data: np.ndarray, lookback: int = 24, horizon: int = 1):
    x, y = [], []
    for i in range(len(data) - lookback - horizon):
        x.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(x), np.array(y)


def main() -> None:
    print("CNN-LSTM time-series demo (synthetic well-log data)\n")
    raw = synthetic_well_log()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)
    x, y = windows(scaled)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = keras.Sequential(
        [
            layers.Input(shape=(x.shape[1], x.shape[2])),
            layers.Conv1D(16, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.LSTM(16),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)

    loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest MSE: {loss:.4f}")
    print("Demo complete — no external dataset required.")


if __name__ == "__main__":
    main()
