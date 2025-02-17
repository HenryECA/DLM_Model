import numpy as np
import matplotlib.pyplot as plt

# Configuración de parámetros
np.random.seed(42)
T = 10  # Número de días (más corto para el caso de uso de acciones)
mu_0 = 100  # Tendencia inicial (precio inicial de la acción)
V = 16  # Varianza del ruido de observación (mayor volatilidad)
W = 4  # Varianza del ruido del sistema (cambios en la tendencia)

# Simulación de precios de acciones reales
precios = []  # Precios observados
tendencia_real = [mu_0]  # Tendencia verdadera
for t in range(1, T + 1):
    mu_t = tendencia_real[-1] + np.random.normal(0, np.sqrt(W))  # Evolución de la tendencia
    precio_t = mu_t + np.random.normal(0, np.sqrt(V))  # Observación con ruido
    tendencia_real.append(mu_t)
    precios.append(precio_t)
tendencia_real = tendencia_real[:-1]  # Ajustar longitud

# Filtro de Kalman
m = [mu_0]  # Estimación inicial de la tendencia
C = [1]  # Varianza inicial de la estimación

parametros_kalman = []  # Almacenar los parámetros en cada paso

for t in range(T):
    # Predicción
    a_t = m[-1]
    R_t = C[-1] + W

    # Observación
    f_t = a_t
    Q_t = R_t + V

    # Ganancia de Kalman
    K_t = R_t / Q_t

    # Actualización
    m_t = a_t + K_t * (precios[t] - f_t)
    C_t = (1 - K_t) * R_t

    m.append(m_t)
    C.append(C_t)

    # Guardar parámetros para análisis
    parametros_kalman.append({
        'Día': t + 1,
        'Predicción a_t': a_t,
        'Varianza Predicción R_t': R_t,
        'Ganancia Kalman K_t': K_t,
        'Actualización m_t': m_t,
        'Varianza Actualización C_t': C_t
    })

# Mostrar parámetros estimados
for params in parametros_kalman:
    print(params)

# Gráfica de resultados
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), precios, label="Precios observados", marker="o", linestyle="none", color="gray")
plt.plot(range(1, T + 1), tendencia_real, label="Tendencia verdadera", linestyle="--", color="blue")
plt.plot(range(1, T + 1), m[1:], label="Tendencia estimada", color="red")
plt.fill_between(range(1, T + 1), 
                 np.array(m[1:]) - 1.96 * np.sqrt(C[1:]), 
                 np.array(m[1:]) + 1.96 * np.sqrt(C[1:]), 
                 color="red", alpha=0.2, label="Intervalo de confianza 95%")
plt.xlabel("Día")
plt.ylabel("Precio de la acción")
plt.title("Estimación de la Tendencia del Precio de Acciones con un DLM")
plt.legend()
plt.show()
