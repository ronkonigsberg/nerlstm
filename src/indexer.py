class Indexer(object):
    def __init__(self):
        self._object_to_index = {}
        self._index_to_object = {}

    def index_object(self, obj):
        if obj in self._object_to_index:
            obj_index = self._object_to_index[obj]
        else:
            obj_index = len(self._object_to_index)
            self._object_to_index[obj] = obj_index
            self._index_to_object[obj_index] = obj
        return obj_index

    def index_object_list(self, object_list):
        for obj in object_list:
            self.index_object(obj)

    def get_index(self, obj):
        return self._object_to_index.get(obj)

    def get_object(self, index):
        return self._index_to_object[index]

    def __len__(self):
        return len(self._object_to_index)
