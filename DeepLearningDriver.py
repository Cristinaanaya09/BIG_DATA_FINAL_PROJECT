from mrjob.job import MRJob
from mrjob.step import MRStep
import os
import random
from decimal import Decimal

class DeepLearningDriver(MRJob):

    def configure_args(self):
        super(DeepLearningDriver, self).configure_args()
        self.add_passthru_arg('--maxEpoch', type=int, help='Maximum number of epochs')
        self.add_passthru_arg('--numLayer', type=int, help='Number of layers')
        self.add_passthru_arg('--numNodeofLayer', type=int, nargs='+', help='Number of nodes in each layer')
        self.add_passthru_arg('--numCase', type=int, help='Number of training items')

    def load_weights_to_cache(self, weight_file):
        # Load weights to distributed cache
        weights = [0.1 * random.gauss(0, 1) for _ in range(self.options.numNodeofLayer[0] * self.options.numNodeofLayer[1])]
        with open(weight_file, 'w') as f:
            f.write(' '.join(map(str, weights)))

        self.add_file_arg('--weights', weight_file)

    def init_weights(self):
        # Initialize weights
        return [0.1 * random.gauss(0, 1) for _ in range(self.options.numNodeofLayer[0] * self.options.numNodeofLayer[1])]

    def run(self):
        weight_file = 'weights.txt'
        self.load_weights_to_cache(weight_file)

        for layer in range(self.options.numLayer - 1):
            # Training phase (RBMMJob)
            weights = self.init_weights()
            for iter in range(1, self.options.maxEpoch + 1):
                weight_file = f'weights-layer-{layer}-iter-{iter}.txt'
                self.load_weights_to_cache(weight_file)

                # Run RBMMJob
                self.mr_job = RBMMJob(args=['-r', 'hadoop', '--weights', weight_file])
                with self.mr_job.make_runner() as runner:
                    runner.run()

                # Update weights based on output

            # Propagation phase (OPTMRJob)
            weight_file = f'weights-layer-{layer}-iter-{self.options.maxEpoch}.txt'
            self.load_weights_to_cache(weight_file)

            # Run OPTMRJob
            self.mr_job = OPTMRJob(args=['-r', 'hadoop', '--weights', weight_file])
            with self.mr_job.make_runner() as runner:
                runner.run()

            # Move to the next layer

if __name__ == '__main__':
    DeepLearningDriver.run()
