import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# datos
X = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [1, 50],
    [7, 85]
])

# etiquetas
y = np.array([0, 0, 1, 1, 0, 1])

# modelo
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compilar modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# entrenar modelo
model.fit(X, y, epochs=100, verbose=0)

# prediccion
nuevo_estudiante = np.array([[5, 75]])
prediccion = model.predict(nuevo_estudiante)

print("Probabilidad de aprobar:", prediccion)
