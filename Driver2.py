import subprocess
import sys

def run_map_reduce(input_path, output_path, mapper_script, reducer_script, weights_file=None):
    hadoop_cmd = [
        'hadoop', 'jar', '/usr/lib/hadoop/hadoop-streaming.jar',
        '-input', input_path,
        '-output', output_path,
        '-mapper', mapper_script,
        '-reducer', reducer_script,
        '-files', f'{mapper_script},{reducer_script}'
    ]

    if weights_file:
        hadoop_cmd.extend(['-cmdenv', f'WEIGHTS_FILE={weights_file}'])

    subprocess.run(hadoop_cmd, check=True)

def main():
    # parse command line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_epoch = int(sys.argv[3])
    num_layer = int(sys.argv[4])
    num_node_of_layer = [int(sys.argv[i]) for i in range(5, 5 + num_layer)]
    num_case = int(sys.argv[5 + num_layer]) if len(sys.argv) > 5 + num_layer else 60000
    epoch_in_mapper = 1

    use_distributed_cache = "1"

    for layer in range(num_layer - 1):
        num_dims = num_node_of_layer[layer]
        num_hid = num_node_of_layer[layer + 1]

        # Initialize weights matrix
        vishid_matrix = [0.1 * random.gauss(0, 1) for _ in range(num_dims * num_hid)]

        for epoch in range(1, max_epoch // epoch_in_mapper + 1):
            weight_file = f'{output_file}/weights/vishid-{layer}-{epoch}.txt'
            current_input = f'{input_file}/{layer}/'
            current_output = f'{output_file}/{layer}/'

            # Configure Hadoop job
            hadoop_conf = {
                'numdims': str(num_dims),
                'numhid': str(num_hid),
                'useDistributedCache': use_distributed_cache
            }

            if use_distributed_cache == "1":
                with open(weight_file, 'w') as f:
                    f.write(' '.join(map(str, vishid_matrix)))

                hadoop_conf['WEIGHTS_FILE'] = weight_file

            run_map_reduce(current_input, current_output, 'RBMMRJobMapper.py', 'RBMMRJobReducer.py', hadoop_conf)

            # Read output files and update weights
            for fs in os.listdir(current_output):
                if not fs.startswith("_"):
                    with open(os.path.join(current_output, fs)) as f:
                        for line in f:
                            tokens = line.strip().split("\t")
                            vishid_matrix[int(tokens[0])] += float(tokens[1]) / num_case

            # Delete temporary output directory
            shutil.rmtree(current_output)

        # Forward propagate to the next layer
        weight_file = f'{output_file}/weights/vishid-{layer}-{max_epoch}.txt'
        current_input = f'{input_file}/{layer}/'
        current_output = f'{input_file}/{layer + 1}/'

        with open(weight_file, 'w') as f:
            f.write(' '.join(map(str, vishid_matrix)))

        hadoop_conf = {
            'numdims': str(num_dims),
            'numhid': str(num_hid),
            'useDistributedCache': use_distributed_cache,
            'WEIGHTS_FILE': weight_file
        }

        run_map_reduce(current_input, current_output, 'OPTMRJobMapper.py', 'OPTMRJobReducer.py', hadoop_conf)

if __name__ == "__main__":
    main()
