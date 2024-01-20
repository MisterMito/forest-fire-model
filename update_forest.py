def update_forest(forest, EMPTY, TREE, FIRE, f, p, size):
    """ Actualizar el estado del bosque para el siguiente paso de tiempo """
    import numpy as np
    new_forest = forest.copy()
    fire_size = 0  # Inicializar el tamaño del incendio

    # Crecimiento probabilístico de nuevos árboles
    growth_sites = (forest == EMPTY) & (np.random.rand(size, size) < p)
    new_forest[growth_sites] = TREE

    # Incendio probabilístico en los árboles
    fire_sites = (forest == TREE) & (np.random.rand(size, size) < f)
    
    # Número de casos en los que hay un incendio aleatorio
    lightning_before = np.sum(new_forest == FIRE)
    new_forest[fire_sites] = FIRE
    lightning_after = np.sum(new_forest == FIRE)
    
    lightning_trees = lightning_after - lightning_before
    
    fire_size += np.sum(fire_sites)  # Contar árboles que se incendiaron

    # Propagación del fuego a árboles vecinos
    for i in range(size):
        for j in range(size):
            if forest[i, j] == TREE:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if 0 <= i + di < size and 0 <= j + dj < size and forest[i + di, j + dj] == FIRE:
                            new_forest[i, j] = FIRE
                            fire_size += 1  # Contar árboles que se incendian por propagación

    # Los árboles quemados se convierten en sitios vacíos
    new_forest[forest == FIRE] = EMPTY

    return new_forest, fire_size, lightning_trees