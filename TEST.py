import numpy as np

# Función para simular la fase de codificación de la RBM
def encode(data, weights, biases):
    # Manejar el desbordamiento usando np.exp con where
    linear_term = data.dot(weights) + biases

    # Aplica min-max scaling a linear_term
    scaled_linear_term = min_max_scaling(linear_term, -1, 1)
    exp_term = np.exp(scaled_linear_term)

    # Calcular la salida de la función sigmoide
    return 1 / (1 + exp_term)


def min_max_scaling(arr, new_min, new_max):
    # Encuentra el mínimo y máximo actual de los valores en el array
    current_min, current_max = np.min(arr), np.max(arr)

    # Realiza el escalado min-max
    scaled_arr = (arr - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    return scaled_arr

# Función para simular la fase de decodificación de la RBM
def decode(hidden, weights, biases):
    # Aplica el min-max scaling inverso al término lineal
    scaled_linear_term = min_max_scaling_inv(hidden, -1, 1)

    # Calcula la salida de la función sigmoide
    return 1 / (1 + np.exp(-(scaled_linear_term.dot(weights.T) + biases)))*25

# Función para realizar el escalado inverso de min-max
def min_max_scaling_inv(arr, new_min, new_max):
    # Encuentra el mínimo y máximo actual de los valores en el array
    current_min, current_max = np.min(arr), np.max(arr)

    # Realiza el escalado inverso de min-max
    scaled_arr_inv = (arr - new_min) / (new_max - new_min) * (current_max - current_min) + current_min

    return scaled_arr_inv



# Función para cargar pesos desde un archivo de texto
import numpy as np

# Función para cargar pesos desde un archivo de texto
def load_weights(file_path):
    with open(file_path, 'r') as file:
        # Lee cada línea y divide en posición y valor de peso
        lines = [line.strip().split('\t') for line in file]

        # Convierte a enteros las posiciones (sin las comillas) y a float los valores de peso
        positions = [int(line[0].strip('"')) for line in lines]
        weights = [float(line[1]) for line in lines]

        # Crea un array de NumPy con los valores de peso ordenados por posición
        sorted_weights = [weight for _, weight in sorted(zip(positions, weights))]
        weights_array = np.array(sorted_weights)

        # Da forma a la matriz de pesos según el tamaño especificado
        weights_matrix = weights_array.reshape(num_visible, num_hidden)

    return weights_matrix



# Función para cargar datos desde un archivo
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = np.loadtxt(file)
    return data

# Función para calcular el error de reconstrucción
def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# Cargar datos y pesos desde los archivos
data_file_path = '/content/MNISTLargo.txt'  # Reemplaza con la ruta correcta

data = load_data(data_file_path)
print("Shape of data:",data.shape)
# Definir parámetros de la RBM (por ejemplo, sesgos)
num_visible = data.shape[1]
num_hidden = 10

weights_file_path = '/content/weights1MNIST.txt'
weights = load_weights(weights_file_path)
print("Shape of weights:", weights.shape)



visible_biases = np.zeros(num_visible)
hidden_biases = np.zeros(num_hidden)

# Simular la fase de codificación y decodificación
encoded_data = encode(data, weights, hidden_biases)
reconstructed_data = decode(encoded_data, weights, visible_biases)

decode(encoded_data, weights, visible_biases)


# Calcular el error de reconstrucción
error = reconstruction_error(data, reconstructed_data)

print(f"Error de reconstrucción: {error}")
