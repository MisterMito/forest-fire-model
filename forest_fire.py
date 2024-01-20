import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from update_forest import update_forest
from scipy.ndimage import label
from scipy import stats


# Parámetros
size = 300  # Tamaño de la cuadrícula
p = 0.01  # Probabilidad de crecimiento de árboles en una celda vacía
f = 0.0001  # Probabilidad de que un árbol se incendie espontáneamente
steps = 10000  # Número de pasos en la simulación



# Estados
EMPTY, TREE, FIRE = 0, 1, 2

# Colores para EMPTY, TREE, y FIRE
colors = ["white", "green", "red"]
cmap = ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5]
norm =  BoundaryNorm(bounds, cmap.N)




def calculate_cluster_stats(forest):
    """ Calcula las variables N(s) y R(s), s, para una configuración dada de forest """
    labeled, num_clusters = label(forest == 1)  # Asumiendo que 1 representa árboles
    cluster_sizes = np.bincount(labeled.ravel())[1:]  # Tamaños de los clústeres
    unique_sizes = np.unique(cluster_sizes)

    # Preparar arrays para estadísticas
    avg_cluster_count = []
    avg_cluster_radius = []

    for size in unique_sizes:
        indices = np.where(cluster_sizes == size)
        num_clusters = len(indices[0])
        avg_cluster_count.append(num_clusters)

        # Calcular el radio de cada clúster
        radii = []
        for idx in indices[0]:
            cluster = np.transpose(np.where(labeled == idx + 1))  # Obtiene las coordenadas del clúster
            center_of_mass = np.mean(cluster, axis=0)
            radii.append(np.sqrt(np.mean(np.sum((cluster - center_of_mass)**2, axis=1))))

        avg_cluster_radius.append(np.mean(radii))

    return unique_sizes, avg_cluster_count, avg_cluster_radius




# Inicializa tus bosques
num_forests = 30  # Número de bosques a simular
forests = [np.zeros((size, size), dtype=int) for _ in range(num_forests)]

# Parámetros de la simulación
num_steps = 400  # Número de pasos en la simulación
record_steps = 10 # Calcular estadísticas cada record_steps pasos

# Arrays para acumular estadísticas
unique_sizes_all = []
avg_cluster_counts_all = []
avg_cluster_radii_all = []

for i in range(np.shape(forests)[0]):
    forest = forests[i]
    for step in range(num_steps):
        forest = np.array(update_forest(forest, EMPTY, TREE, FIRE, f, p, size)[0])
        if step % record_steps == 0:  # Calcular estadísticas cada 10 pasos
            unique_sizes, avg_cluster_counts, avg_cluster_radii = calculate_cluster_stats(forest)
            
            # Acumular estadísticas
            unique_sizes_all.extend(unique_sizes)
            avg_cluster_counts_all.extend(avg_cluster_counts)
            avg_cluster_radii_all.extend(avg_cluster_radii)
            
unique_sizes = np.array(unique_sizes_all)
counts = np.array(avg_cluster_counts_all)
radii = np.array(avg_cluster_radii_all)

# Diccionarios para acumular sumas y conteos
sum_counts = {}
sum_radii = {}
count_sizes = {}

# Acumular sumas y conteos
for size, count, radius in zip(unique_sizes, counts, radii):
    if size not in sum_counts:
        sum_counts[size] = 0
        sum_radii[size] = 0
        count_sizes[size] = 0
    sum_counts[size] += count
    sum_radii[size] += radius
    count_sizes[size] += 1

# Calcular promedios
avg_counts = {size: sum_counts[size] / count_sizes[size] for size in sum_counts}
avg_radii = {size: sum_radii[size] / count_sizes[size] for size in sum_radii}

# Convertir a listas para gráficos u otro análisis
final_unique_sizes = list(avg_counts.keys())
final_avg_counts = list(avg_counts.values())
final_avg_radii = list(avg_radii.values())


unique_sizes = np.array(final_unique_sizes)

counts = np.array(final_avg_counts)

radii = np.array(final_avg_radii)


# Filtrar para eliminar los valores donde la función de escala lo haga cero

mask = np.log10(unique_sizes) <= 1.7
unique_sizes_fit = unique_sizes[mask]
counts_fit = counts[mask]
radii_fit = radii[mask]


# Convertir a logarítmica
log_sizes = np.log10(unique_sizes)
log_counts = np.log10(counts)
log_radii = np.log10(radii)


log_sizes_fit = np.log10(unique_sizes_fit)
log_counts_fit = np.log10(counts_fit)
log_radii_fit = np.log10(radii_fit)


# Eliminar los valores infinitos en log_radii y sus correspondientes en log_sizes y log_counts
valid = ~np.isinf(log_radii)
log_sizes_filtered = log_sizes[valid]
log_counts_filtered = log_counts[valid]
log_radii_filtered = log_radii[valid]

valid_fit = ~np.isinf(log_radii_fit)
log_sizes_fit_filtered = log_sizes_fit[valid_fit]
log_counts_fit_filtered = log_counts_fit[valid_fit]
log_radii_fit_filtered = log_radii_fit[valid_fit]

# Ajuste lineal para counts

slope_counts, intercept_counts, r_value_counts, p_value_counts, std_err_counts = stats.linregress(log_sizes_fit_filtered, log_counts_fit_filtered)
r_squared_counts = r_value_counts ** 2


# Ajuste lineal para radii
slope_radii, intercept_radii, r_value_radii, p_value_radii, std_err_radii = stats.linregress(log_sizes_fit_filtered, log_radii_fit_filtered)
r_squared_radii = r_value_radii ** 2
# Gráfico

fig, ax1 = plt.subplots(figsize=(10, 7))

# Graficar los datos y el ajuste lineal para N(s)
ax1.plot(log_sizes_filtered, log_counts_filtered, 'd', label='Data ' + r'$N(s)$')
ax1.plot(log_sizes_fit_filtered, slope_counts*log_sizes_fit_filtered + intercept_counts, label=f'Linear Fit (Slope = {slope_counts:.2f})')
ax1.set_xlabel(r'$\log{(s)}$')
ax1.set_ylabel(r'$\log{(N(s))}$')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')
ax1.minorticks_on()

# Crear un segundo eje que comparte el eje x con ax1
ax2 = ax1.twinx()
ax2.plot(log_sizes_filtered, log_radii_filtered, 'h', label='Data ' +  r'$R(s)$', color='tab:orange')
ax2.plot(log_sizes_fit_filtered, slope_radii*log_sizes_fit_filtered + intercept_radii, label=f'Linear Fit (Slope = {slope_radii:.2f})', color='tab:red')
ax2.set_ylabel(r'$\log{(R(s))}$')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')
ax2.minorticks_on()


plt.tight_layout()
plt.show()

print(f"Ajuste lineal para counts: y = {slope_counts:.4f}x + {intercept_counts:.4f}")
print(f'R^2 = {r_squared_counts:.4f}')
print(f"Ajuste lineal para radii: y = {slope_radii:.4f}x + {intercept_radii:.4f}")
print(f'R^2 = {r_squared_radii:.4f}')



