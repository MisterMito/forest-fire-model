import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from update_forest import update_forest


p = 0.01 # Probabilidad de crecimiento de árboles en una celda vacía
f = 0.0001 # Probabilidad de que un árbol se incendie espontáneamente
# Parámetros
size = 300  # Tamaño de la cuadrícula



# Estados
EMPTY, TREE, FIRE = 0, 1, 2

# Colores para EMPTY, TREE, y FIRE
colors = ["white", "green", "red"]
cmap = ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5]
norm =  BoundaryNorm(bounds, cmap.N)




# Inicializar la cuadrícula
forest = np.zeros((size, size), dtype=int)


# Función de animación
def animate(frame):
    global forest
    new_forest, fire_size = update_forest(forest, EMPTY, TREE, FIRE, f, p, size)
    forest = new_forest
    
    ax.imshow(forest, cmap=cmap, norm=norm)
    ax.set_title('Forest evolution for p/f = %i at ' %(p/f) + f"step: {frame}")
    ax.set_xticks([])  # Eliminar las marcas del eje X
    ax.set_yticks([])  # Eliminar las marcas del eje Y
    

# Crear figura y ejes
fig, ax = plt.subplots()

# Crear animación
ani = animation.FuncAnimation(fig, animate, frames=200, interval=100)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Empty', 'Tree', 'Fire'])  # etiquetas para EMPTY, TREE, FIRE

# To display the animation, we need to use a specific matplotlib backend
plt.rcParams['animation.html'] = 'html5'



# To save the animation, we need to use a writer
from matplotlib.animation import PillowWriter

# Saving the animation as a GIF file
gif_filename = "/Users/sicilia/Documents/Física/Master/1er semestre/Fenomenos cooperativos/Percolation/forest_fire_animation_300_300.gif"
writer = PillowWriter(fps=20)  # frames per second
ani.save(gif_filename, writer=writer)

