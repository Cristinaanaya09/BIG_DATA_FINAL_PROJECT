from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import numpy as np

class OPTMRJobMapper(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def configure_args(self):
        super(OPTMRJobMapper, self).configure_args()
        self.add_file_arg('--weights', help='Path to the weights file')

    def mapper(self, _, line):
        # Parse input data
        input_data = [list(map(int, line.strip().split())) for line in data_str.split('\n')]
        numdims = len(input_data)

        # Read weights from the distributed cache
        weights_file = self.options.weights
        with open(weights_file, 'r') as f:
            weights_line = f.readline().strip().split()
            vishid = np.array(list(map(float, weights_line)))

        # Initialize parameters
        numhid = len(vishid) // numdims
        hidbiases = np.zeros(numhid)
        visbiases = np.zeros(numdims)

        # Perform forward propagation
        poshidprobs = np.dot(input_data, vishid).reshape(1, numhid) + hidbiases
        poshidstates = (np.random.rand(1, numhid) < 1 / (1 + np.exp(-poshidprobs))).astype(float)

        # Output the key and updated values
        updated_values = ' '.join(map(str, poshidstates.flatten().astype(int)))
        yield None, updated_values

if __name__ == '__main__':
    OPTMRJobMapper.run()
