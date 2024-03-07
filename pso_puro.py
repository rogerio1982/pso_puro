import numpy as np

# Definir a função de Rosenbrock para otimização
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Definir a função PSO
def pso(func, n_particles, n_dimensions, n_iterations):
    # Inicializar as posições e velocidades das partículas
    positions = np.random.uniform(-5, 5, (n_particles, n_dimensions))
    velocities = np.random.uniform(-1, 1, (n_particles, n_dimensions))

    # Inicializar a melhor posição global e o melhor valor global
    best_global_position = np.copy(positions[0])
    best_global_value = func(best_global_position)

    # Inicializar a melhor posição local e os melhores valores locais
    best_local_positions = np.copy(positions)
    best_local_values = np.array([func(pos) for pos in positions])

    # Iterar através das iterações do PSO
    for _ in range(n_iterations):
        # Atualizar as melhores posições locais
        for i in range(n_particles):
            if func(positions[i]) < best_local_values[i]:
                best_local_values[i] = func(positions[i])
                best_local_positions[i] = np.copy(positions[i])

        # Atualizar a melhor posição global
        min_index = np.argmin(best_local_values)
        if best_local_values[min_index] < best_global_value:
            best_global_value = best_local_values[min_index]
            best_global_position = np.copy(best_local_positions[min_index])

        # Atualizar as velocidades e posições das partículas
        w = 0.5  # inércia
        c1 = 1.5  # fator cognitivo
        c2 = 1.5  # fator social
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)

        velocities = w * velocities + c1 * r1 * (best_local_positions - positions) + c2 * r2 * (best_global_position - positions)
        positions = positions + velocities

    return best_global_position, best_global_value

if __name__ == "__main__":
    # Parâmetros do PSO
    n_particles = 20
    n_dimensions = 5
    n_iterations = 100

    # Executar o PSO na função de Rosenbrock
    best_solution, best_value = pso(rosenbrock, n_particles, n_dimensions, n_iterations)

    print("Melhor solução encontrada:", best_solution)
    print("Melhor valor encontrado:", best_value)
