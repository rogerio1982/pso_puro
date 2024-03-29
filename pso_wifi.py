import numpy as np
import random

# Parâmetros do PSO
num_particles = 20
max_iter = 50
c1 = 2
c2 = 2
w = 0.5

# Parâmetros do problema
num_users = 100
num_access_points = 3
area_width = 100
area_height = 100

# Função de avaliação (fitness)
def evaluate_solution(solution):
    # Calcular a qualidade da solução (pontos de acesso)
    # Aqui, a qualidade pode ser medida pela cobertura do sinal para os usuários
    # Quanto mais usuários estiverem dentro da área de cobertura dos pontos de acesso, melhor
    coverage = 0
    for user in users:
        for access_point in solution:
            distance = np.linalg.norm(user - access_point)
            if distance < 10:  # Assumindo um raio de cobertura de 10 unidades
                coverage += 1
                break
    return coverage

# Inicialização dos usuários em posições aleatórias
users = np.random.rand(num_users, 2) * area_width

# Função de inicialização das partículas
def initialize_particles():
    particles = []
    for _ in range(num_particles):
        particle = np.random.rand(num_access_points, 2) * area_width
        particles.append(particle)
    return particles

# Algoritmo PSO
def pso():
    particles = initialize_particles()
    global_best_position = None
    global_best_fitness = -1

    for _ in range(max_iter):
        for particle in particles:
            fitness = evaluate_solution(particle)
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle

        for particle in particles:
            # Atualizar velocidades e posições das partículas
            velocity = w * particle + c1 * random.random() * (global_best_position - particle) + c2 * random.random() * (global_best_position - particle)
            particle += velocity

    return global_best_position

# Executar o algoritmo PSO
best_solution = pso()
print("Melhor solução encontrada:")
print(best_solution)


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do problema
num_users = 100
num_access_points = 3
area_width = 100
area_height = 100

# Inicialização dos usuários em posições aleatórias
users = np.random.rand(num_users, 2) * area_width

# Melhor solução encontrada pelo PSO (posição dos pontos de acesso)
best_solution = np.array([[30, 30], [70, 70], [50, 20]])  # Exemplo de posição dos pontos de acesso

# Plotar os usuários
plt.scatter(users[:, 0], users[:, 1], color='blue', label='Usuários')

# Plotar os pontos de acesso
plt.scatter(best_solution[:, 0], best_solution[:, 1], color='red', marker='s', label='Pontos de Acesso')

# Plotar os raios de cobertura dos pontos de acesso
for point in best_solution:
    circle = plt.Circle((point[0], point[1]), 10, color='red', fill=False)  # Raio de cobertura de 10 unidades
    plt.gca().add_artist(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Posições dos Pontos de Acesso e Usuários')
plt.legend()
plt.grid(True)
plt.xlim(0, area_width)
plt.ylim(0, area_height)
plt.show()
