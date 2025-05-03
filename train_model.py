import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import preprocess as pp

# 2 Keras model 개발
X_train, X_valid, y_train, y_valid = train_test_split(
    pp.fires_prepared, pp.fires_labels, test_size=0.2, random_state=42)
X_test, y_test = pp.fires_test_prepared, pp.fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

model.save("fires_model.keras")

X_new = X_test[:3]
print("\n예측 결과:\n", np.round(model.predict(X_new), 2))
import joblib
joblib.dump(pp.full_pipeline, "full_pipeline.pkl")