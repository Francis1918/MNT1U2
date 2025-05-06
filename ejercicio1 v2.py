import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import matplotlib
matplotlib.use('TkAgg')  # Backend compatible con animaciones

# Definimos el polinomio y su derivada
def f(x):
    return x**4 + 540*x**3 + 109124*x**2 + 9781632*x + 328188672

def f_prime(x):
    return 4*x**3 + 3*540*x**2 + 2*109124*x + 9781632

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
x0 = -500  # Punto inicial
solucion, iteraciones = newton_raphson(x0)

# Crear datos para graficar el polinomio
x_vals = np.linspace(-600, 100, 1000)
y_vals = f(x_vals)

# Configurar la gráfica
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_vals, label="Polinomio", color="blue")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
point, = ax.plot([], [], "ro", markersize=8, label="Iteración")

# Límites para ver mejor la raíz
ax.set_xlim(-550, 0)
ax.set_ylim(-1e10, 1e10)

# Añadir texto para mostrar el valor actual
texto = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animación
def update(frame):
    x = iteraciones[frame]
    y = f(x)
    point.set_data([x], [y])
    # Actualizar texto con la iteración actual
    texto.set_text(f'Iteración {frame}: x = {x:.6f}, f(x) = {y:.6f}')
    return point, texto

ani = FuncAnimation(fig, update, frames=len(iteraciones), interval=500, repeat=True, blit=True)

# Configurar para visualización HTML
from matplotlib import rc
rc('animation', html='jshtml')

# Guardar la animación
ani.save('newton_raphson.gif', writer='pillow', fps=2)

# Mostrar la gráfica
ax.legend()
plt.title("Método de Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.savefig('grafico_final.png')

# Mostrar animación interactiva usando IPython.display
animation_html = ani.to_jshtml()
display(HTML(animation_html))

# También mostrar la gráfica estática con la solución final
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_vals, y_vals, label="Polinomio", color="blue")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.plot(solucion, f(solucion), "ro", markersize=8)
ax2.annotate(f'Solución: x = {solucion:.6f}',
             xy=(solucion, f(solucion)),
             xytext=(solucion-50, f(solucion)+2e9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
ax2.set_xlim(-550, 0)
ax2.set_ylim(-1e10, 1e10)
ax2.legend()
plt.title("Método de Newton-Raphson: Solución Final")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
display(fig2)

print("Animación guardada como 'newton_raphson.gif'")
print(f"La solución encontrada es x = {solucion:.6f}")