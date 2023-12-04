import numpy as np

def encode(data, weights, biases):
    
    linear_term = data.dot(weights) + biases
    scaled_linear_term = min_max_scaling(linear_term, -1, 1)
    exp_term = np.exp(scaled_linear_term)

    return 1 / (1 + exp_term)


def min_max_scaling(arr, new_min, new_max):
    current_min, current_max = np.min(arr), np.max(arr)
    scaled_arr = (arr - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    return scaled_arr


def decode(hidden, weights, biases):
    
    scaled_linear_term = min_max_scaling_inv(hidden, -1, 1)
    return 1 / (1 + np.exp(-(scaled_linear_term.dot(weights.T) + biases)))*25

def min_max_scaling_inv(arr, new_min, new_max):
    
    current_min, current_max = np.min(arr), np.max(arr)
    scaled_arr_inv = (arr - new_min) / (new_max - new_min) * (current_max - current_min) + current_min

    return scaled_arr_inv

def load_weights(file_path):
    with open(file_path, 'r') as file:
        
        lines = [line.strip().split('\t') for line in file]

        positions = [int(line[0].strip('"')) for line in lines]
        weights = [float(line[1]) for line in lines]
        sorted_weights = [weight for _, weight in sorted(zip(positions, weights))]
        weights_array = np.array(sorted_weights)
        weights_matrix = weights_array.reshape(num_visible, num_hidden)

    return weights_matrix

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = np.loadtxt(file)
    return data

def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

data_file_path = '/content/MNISTLargo.txt' 
data = load_data(data_file_path)
print("Shape of data:",data.shape)

num_visible = data.shape[1]
num_hidden = 10

weights_file_path = '/content/weights1MNIST.txt'
weights = load_weights(weights_file_path)
print("Shape of weights:", weights.shape)

visible_biases = np.zeros(num_visible)
hidden_biases = np.zeros(num_hidden)

encoded_data = encode(data, weights, hidden_biases)
reconstructed_data = decode(encoded_data, weights, visible_biases)

decode(encoded_data, weights, visible_biases)

error = reconstruction_error(data, reconstructed_data)
print(f"Reconstruction error: {error}")
