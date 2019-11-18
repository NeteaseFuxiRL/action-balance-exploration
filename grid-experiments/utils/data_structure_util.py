from queue import Queue
import random
import pickle


class FixSizeQueue(object):
    def __init__(self, n):
        self.data = []
        self.maxsize = n

    def enqueue(self, a):
        if self.size() >= self.maxsize:
            self.data.pop()
        self.data.insert(0, a)

    def dequeue(self):
        return self.data.pop()

    def rear(self, n=1):
        return self.data[-n]

    def front(self, n=0):
        return self.data[n]

    def size(self):
        return len(self.data)

    def is_full(self):
        return self.size() >= self.maxsize

    def is_empty(self):
        return self.size() == 0

    def mean(self):
        if self.size() == 0:
            return None
        else:
            return float(sum(self.data)) / self.size()


class ObjectPool(object):
    """
    Object pool
    """

    def __init__(self, num):
        self.maxsize = num
        self.object_queue = Queue(self.maxsize)
        # self.object_queue = multiprocessing.Manager().Queue(self.maxsize)

    def size(self):
        return self.object_queue.qsize()

    def acquire(self):
        return self.object_queue.get()

    def release(self, obj):
        self.object_queue.put(obj)

    def empty(self):
        return self.object_queue.empty()

    def full(self):
        return self.object_queue.full()


class FixSizeHeap(object):
    def __init__(self, n):
        self._storage = []
        self.max_size = n
        self.explore_prob = 0.5

    def put(self, data):
        if not self.is_full() and data not in self._storage:
            self._storage.append(data)

    def get(self):
        if self.is_empty():
            return None
        else:
            if not self.is_full() and random.random() < self.explore_prob:
                return None
            else:
                index = random.randint(0, len(self._storage) - 1)
                return self._storage[index]

    def size(self):
        return len(self._storage)

    def is_full(self):
        return self.size() >= self.max_size

    def is_empty(self):
        return self.size() == 0

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._storage, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data_tmp = pickle.load(f)
        self._storage = data_tmp[:self.max_size]


class StatisticInfo(object):
    """
    Used for custom sort. `Name` can be seen as id to check uniqueness, and sort by other attributes.
    """

    def __init__(self, name, statistic_windows=1):
        self.name = name
        self.records = FixSizeQueue(statistic_windows)
        self.win_rate = 0.

    def __repr__(self):
        return '{}_{}'.format(self.name, self.win_rate)

    def add(self, s):
        self.records.enqueue(s)
        self.update_win_rate()

        return self.win_rate

    def update_win_rate(self):
        self.win_rate = float(sum(self.records.data)) / self.records.size()

        return self.win_rate

    def get_win_rate(self):
        return self.win_rate

    def is_full(self):
        return self.records.is_full()

    def ready_for_eval(self):
        return self.is_full()
