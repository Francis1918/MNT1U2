import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Usar un backend compatible

# Definimos la función y su derivada
def f(x):
    return np.sin(x) / x - 0.015

def f_prime(x):
    return (x * np.cos(x) - np.sin(x)) / (x**2)

# Método de Newton-Raphson
def newton_raphson(x0, tol=1e-6, max_iter=100):
    x = x0
    iteraciones = [x]
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-12:  # Evitar división por cero
            break
        x_new = x - fx / fpx
        iteraciones.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x, iteraciones

# Configuración inicial
x0 = 2  # Punto inicial
solucion, iteraciones = newton_raphson(x0)

# Crear datos para graficar la función
x_vals = np.linspace(1, 10, 1000)
y_vals = np.sin(x_vals) / x_vals

# Configurar la gráfica
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_vals, label="f(x) = sin(x)/x", color="blue")
ax.axhline(0.015, color="red", linewidth=0.8, linestyle="--", label="y = 0.015")
point, = ax.plot([], [], "ro", markersize=8, label="Iteración")

# Añadir texto para mostrar el valor actual
texto = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animación
def update(frame):
    x = iteraciones[frame]
    y = np.sin(x) / x
    # Convertir a listas para evitar el error 'x must be a sequence'
    point.set_data([x], [y])
    # Actualizar texto con la iteración actual
    texto.set_text(f'Iteración {frame}: x = {x:.6f}, f(x) = {y:.6f}')
    return point, texto

ani = FuncAnimation(fig, update, frames=len(iteraciones), interval=500, repeat=True, blit=True)

# Guardar la animación directamente a un archivo
ani.save('newton_raphson_sin_x.gif', writer='pillow', fps=2)

# Mostrar la gráfica
ax.legend()
plt.title("Método de Newton-Raphson para f(x) = sin(x)/x - 0.015 = 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)

# En lugar de plt.show(), guardar y abrir
plt.tight_layout()
plt.savefig('grafico_final_sin_x.png')

print("Animación guardada como 'newton_raphson_sin_x.gif'")
print(f"El valor xT donde f(x) < 0.015 para todo x ≥ xT es: {solucion:.6f}")