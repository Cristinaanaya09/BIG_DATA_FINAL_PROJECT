from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import numpy as np
from mrjob.protocol import JSONProtocol



class RBMMRJobMapper(MRJob):

    OUTPUT_PROTOCOL = JSONProtocol

    def __init__(self, *args, **kwargs):
        super(RBMMRJobMapper, self).__init__(*args, **kwargs)
        self.numdims = None
        self.numhid = None

    def configure_args(self):
        super(RBMMRJobMapper, self).configure_args()
        self.add_passthru_arg('--numdims', type=int, help='Number of nodes in the input layer')
        self.add_passthru_arg('--numhid', type=int, help='Number of nodes in the hidden layer')

    def initialize(self):
        self.epsilonw = 0.1
        self.weightcost = 0.000
        self.numhid = self.options.numhid
        self.numdims = self.options.numdims
        self.vishid = np.random.rand(self.numdims, self.numhid) * 0.01
        self.vishidinc = np.zeros((self.numdims, self.numhid))

    def getposphase(self, data):
        # Perform positive phase calculations
        poshidprobs = np.dot(data, self.vishid) #W.T(pesos)*y(data)
        poshidprobs += np.zeros(poshidprobs.shape)
        poshidprobs = 1 / (1 + np.exp(-poshidprobs)) #SIGMOID, 1/(1+e^-x)
        posprods = np.dot(data.reshape(-1, 1), poshidprobs.reshape(1, -1))

        # Ensure poshidstates has the right shape
        poshidstates = (np.random.rand(self.numhid) < poshidprobs.flatten()).astype(float)

        return posprods, poshidstates.reshape(1, -1)



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

    def mapper(self, _, line):
        # Map function
        

        self.initialize()

        input_data = list(map(int, line.strip().split()))
        data = np.array(input_data[:self.numdims]) / 255.0

        posprods, poshidstates = self.getposphase(data)
        negprods = self.getnegphase(poshidstates)
        self.update(posprods, negprods)

        # Output the weight updates
        for i in range(self.numdims):
            for j in range(self.numhid):
                key = f"{i * self.numhid + j}"
                value = str(float(self.vishidinc[i, j]))
                yield key, value

if __name__ == '__main__':
    RBMMRJobMapper.run()
