from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np

class RBMMRJob(MRJob):

    def configure_args(self):
        super(RBMMRJob, self).configure_args()
        self.add_passthru_arg('--numdims', type=int, help='Number of nodes in the input layer')
        self.add_passthru_arg('--numhid', type=int, help='Number of nodes in the hidden layer')

    def steps(self):
        return [
            MRStep(mapper_init=self.configure,
                   mapper=self.map,
                   reducer=self.reduce)
        ]

    def initialize(self):
        # Initialize parameters and variables
        # Note: In MRJob, we don't maintain state between different mapper calls, so some modifications are made
        self.epsilonw = 0.1
        self.weightcost = 0.000
        self.numhid = self.options.numhid
        self.numdims = self.options.numdims
        self.vishidinc = np.zeros((self.numdims, self.numhid))

    def getposphase(self, data):
        # Perform positive phase calculations
        poshidprobs = np.dot(data, self.vishid)
        poshidprobs += np.zeros((1, self.numhid))
        poshidprobs = 1 / (1 + np.exp(-poshidprobs))
        posprods = np.dot(data.T, poshidprobs)
        poshidstates = (np.random.rand(1, self.numhid) < poshidprobs).astype(float)
        return posprods, poshidstates

    def getnegphase(self, poshidstates):
        # Perform negative phase calculations
        negdata = np.dot(poshidstates, self.vishid.T)
        negdata += np.zeros((1, self.numdims))
        negdata = 1 / (1 + np.exp(-negdata))
        neghidprobs = np.dot(negdata, self.vishid)
        neghidprobs += np.zeros((1, self.numhid))
        neghidprobs = 1 / (1 + np.exp(-neghidprobs))
        negprods = np.dot(negdata.T, neghidprobs)
        return negprods

    def update(self, posprods, negprods):
        # Perform weight update
        momentum = 0.5  # Assuming a constant momentum value
        self.vishidinc = momentum * self.vishidinc + self.epsilonw * ((posprods - negprods) - self.weightcost * self.vishid)

    def map(self, _, line):
        # Map function
        input_data = list(map(int, line.strip().split()))
        data = np.array(input_data[:self.numdims]) / 255.0
        self.initialize()
        posprods, poshidstates = self.getposphase(data)
        negprods = self.getnegphase(poshidstates)
        self.update(posprods, negprods)

        # Output the weight updates
        for i in range(self.numdims):
            for j in range(self.numhid):
                yield f"{i * self.numhid + j}", float(self.vishidinc[i, j])

    def reduce(self, key, values):
        sum_updates = sum(values)
        yield key, sum_updates

if __name__ == '__main__':
    MRJobJob = RBMMRJob(args=['input_file.txt', '--numdims', 'your_numdims_value', '--numhid', 'your_numhid_value'])
    with MRJobJob.make_runner() as runner:
        runner.run()
