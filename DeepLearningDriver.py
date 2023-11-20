import subprocess
import os
import random

def load_weights_to_cache(weight_file):
    # Load weights to distributed cache
    weights = [0.1 * random.gauss(0, 1) for _ in range(numNodeofLayer[0] * numNodeofLayer[1])]
    with open(weight_file, 'w') as f:
        f.write(' '.join(map(str, weights)))

    return weight_file

def init_weights():
    # Initialize weights
    return [0.1 * random.gauss(0, 1) for _ in range(numNodeofLayer[0] * numNodeofLayer[1])]

def run_map_reduce(input_path, output_path, mapper_class, reducer_class, weights_file=None):
    # Construct Hadoop Streaming command
    hadoop_cmd = [
        'hadoop', 'jar', '/path/to/hadoop-streaming.jar',
        '-input', input_path,
        '-output', output_path,
        '-mapper', mapper_class,
        '-reducer', reducer_class,
        '-file', mapper_class,
        '-file', reducer_class
    ]

    if weights_file:
        hadoop_cmd.extend(['-cmdenv', f'WEIGHTS_FILE={weights_file}'])

    subprocess.run(hadoop_cmd, check=True)

if __name__ == '__main__':
    # Argument format: /input_file/ /output_file/ maxiter layernum nodeNumofLayer1 nodeNumofLayer2 ...
    input_file = '/path/to/input'
    output_file = '/path/to/output'
    max_epoch = 10
    num_layer = 3
    numNodeofLayer = [input() for _ in range(num_layer)]

    num_case = 60000

    epoch_in_mapper = 1

    use_distributed_cache = "1"

    for layer in range(num_layer - 1):
        # Training phase (RBMMJob)
        weights = init_weights()
        for iter in range(1, max_epoch + 1):
            weight_file = f'weights-layer-{layer}-iter-{iter}.txt'
            load_weights_to_cache(weight_file)

            # Run RBMMJob
            run_map_reduce(input_file, output_file, 'RBMMapper.py', 'RBMReducer.py', weights_file=weight_file)

            # Update weights based on output

        # Propagation phase (OPTMRJob)
        weight_file = f'weights-layer-{layer}-iter-{max_epoch}.txt'
        load_weights_to_cache(weight_file)

        # Run OPTMRJob
        run_map_reduce(input_file, output_file, 'OptMapper.py', 'OptReducer.py', weights_file=weight_file)

        # Move to the next layer
