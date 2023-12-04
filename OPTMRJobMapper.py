import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol

def min_max_scaling(arr, new_min, new_max):
    
    current_min, current_max = np.min(arr), np.max(arr)
    scaled_arr = (arr - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    return scaled_arr

class OPTMRJobMapper(MRJob):
    def configure_args(self):
        super(OPTMRJobMapper, self).configure_args()
        self.add_passthru_arg('--numdims', type=int, help='Number of dimensions')
        self.add_passthru_arg('--numhid', type=int, help='Number of hidden units')
        self.add_file_arg('--weights', help='Path to weights file')

    def mapper_init(self):
        
        self.epsilonw = 0.1
        self.epsilonvb = 0.1
        self.espilonhb = 0.1
        self.weightcost = 0.000
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9
        self.numhid = self.options.numhid
        self.numdims = self.options.numdims

        self.data = np.zeros(self.numdims)
        self.hidbiases = np.zeros(self.numhid)
        self.visbiases = np.zeros(self.numdims)
        self.poshidprobs = np.zeros(self.numhid)
        self.neghidprobs = np.zeros(self.numhid)
        self.posprods = np.zeros((self.numdims, self.numhid))
        self.negprods = np.zeros((self.numdims, self.numhid))
        self.vishidinc = np.zeros((self.numdims, self.numhid))

        with open(self.options.weights, 'r') as weights_file:
            weight_list = []
            for line in weights_file:
                tokens = line.strip().split('\t')
                if len(tokens) == 2:
                    feature_index = int(tokens[0].strip('"'))
                    weight_value = float(tokens[1])
                    weight_list.append((feature_index, weight_value))

            weight_list.sort(key=lambda x: x[0])
            sorted_weights = [weight_value for _, weight_value in weight_list]
            self.vishid = np.array(sorted_weights).reshape(self.numdims, self.numhid)
    

    def prop2next_layer(self):
        self.poshidprobs = self.data.dot(self.vishid) + self.hidbiases  # y*W + b
        scaled_linear_term = min_max_scaling(self.poshidprobs, -1, 1)
        self.poshidprobs = 1 / (1 + np.exp(-scaled_linear_term))  # SIGMOID: Data between 0 and 1

    def mapper(self, _, line):
        self.data = np.array(list(map(int, line.strip().split())))
        self.prop2next_layer()
        updated = self.poshidprobs.tolist()

        yield None, updated

if __name__ == '__main__':
    OPTMRJobMapper.OUTPUT_PROTOCOL = JSONProtocol
    OPTMRJobMapper.run()

