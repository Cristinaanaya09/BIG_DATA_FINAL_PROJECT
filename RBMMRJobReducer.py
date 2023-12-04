from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
from mrjob.protocol import JSONProtocol


class RBMMRJobReducer(MRJob):
    INPUT_PROTOCOL = JSONProtocol
    def reducer(self, key, values):
        sum_updates = sum(float(v) for v in values)
        yield key, sum_updates

if __name__ == '__main__':
    RBMMRJobReducer.run()
