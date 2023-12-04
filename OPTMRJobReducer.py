from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

class OPTMRJobReducer(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def reducer(self, key, values):
        for value in values:
            # Directly output the <key, value> pair
            yield key, value

if __name__ == '__main__':
    OPTMRJobReducer.run()