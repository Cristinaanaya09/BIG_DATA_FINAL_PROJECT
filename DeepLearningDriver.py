import subprocess
import os
import random
import sys

def load_weights_to_cache(weight_file, num_node_of_layer):
    # Load weights to distributed cache
    weights = [0.1 * random.gauss(0, 1) for _ in range(num_node_of_layer[0] * num_node_of_layer[1])]
    with open(weight_file, 'w') as f:
        f.write(' '.join(map(str, weights)))

    return weight_file

def init_weights(num_node_of_layer):
    # Initialize weights
    return [0.1 * random.gauss(0, 1) for _ in range(num_node_of_layer[0] * num_node_of_layer[1])]

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
    if len(sys.argv) < 7:
        print("Usage: python script.py /path/to/input /path/to/output maxiter layernum nodeNumofLayer1 nodeNumofLayer2 ...")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_epoch = int(sys.argv[3])
    num_layer = int(sys.argv[4])
    num_node_of_layer = [int(sys.argv[i]) for i in range(5, len(sys.argv))]

    num_case = 60000
    epoch_in_mapper = 1
    use_distributed_cache = "1"

    for layer in range(num_layer - 1):
        # Training phase (RBMMJob)
        weights = init_weights(num_node_of_layer)
        for iter in range(1, max_epoch + 1):
            weight_file = f'weights-layer-{layer}-iter-{iter}.txt'
            load_weights_to_cache(weight_file, num_node_of_layer)

            # Run RBMMJob
            run_map_reduce(input_file, output_file, 'RBMMJob.py', weights_file=weight_file)
            
            # Update weights based on output
            
        # Propagation phase (OPTMRJob)
        weight_file = f'weights-layer-{layer}-iter-{max_epoch}.txt'
        load_weights_to_cache(weight_file, num_node_of_layer)
        
        # Run OPTMRJob (assuming OPTMRJob.py contains both mapper and reducer logic)
        run_map_reduce(input_file, output_file, 'OPTMRJob.py', weights_file=weight_file)

        # Move to the next layer

