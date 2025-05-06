import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import matplotlib
matplotlib.use('TkAgg')

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

# Mejorar la visualización
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 10)
ax.set_ylim(-0.05, 0.2)

# Añadir texto para mostrar el valor actual
texto = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animación
def update(frame):
    x = iteraciones[frame]
    y = np.sin(x) / x
    point.set_data([x], [y])
    texto.set_text(f'Iteración {frame}: x = {x:.6f}, f(x) = {y:.6f}')
    return point, texto

ani = FuncAnimation(fig, update, frames=len(iteraciones), interval=500, repeat=True, blit=True)

# Guardar la animación en formato compatible con HTML
from matplotlib import rc
rc('animation', html='jshtml')

# Mostrar la gráfica
ax.legend()
plt.title("Método de Newton-Raphson para f(x) = sin(x)/x - 0.015 = 0")
plt.xlabel("x")
plt.ylabel("f(x)")

# Guardar la animación para visualización fuera del notebook
ani.save('newton_raphson_sin_x.gif', writer='pillow', fps=2)

# Usar IPython.display para mostrar la animación interactiva en el notebook
animation_html = ani.to_jshtml()
display(HTML(animation_html))

# También mostrar la gráfica estática con la solución final
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_vals, y_vals, label="f(x) = sin(x)/x", color="blue")
ax2.axhline(0.015, color="red", linewidth=0.8, linestyle="--", label="y = 0.015")
ax2.plot(solucion, np.sin(solucion)/solucion, "ro", markersize=8)
ax2.annotate(f'Solución: x = {solucion:.6f}',
             xy=(solucion, np.sin(solucion)/solucion),
             xytext=(solucion+1, np.sin(solucion)/solucion+0.02),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 10)
ax2.set_ylim(-0.05, 0.2)
ax2.legend()
plt.title("Solución: Valor xT donde f(x) < 0.015 para todo x ≥ xT")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.savefig('solucion_final_sin_x.png')
display(fig2)

print("Animación guardada como 'newton_raphson_sin_x.gif'")
print(f"El valor xT donde f(x) < 0.015 para todo x ≥ xT es: {solucion:.6f}")
print("Gráfica de solución guardada como 'solucion_final_sin_x.png'")