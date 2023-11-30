from mrjob.job import MRJob

class RBMMRJobReducer(MRJob):

    def reducer(self, key, values):
        sum_updates = sum(values)
        yield key, sum_updates

if __name__ == '__main__':
    RBMMRJobReducer.run()
