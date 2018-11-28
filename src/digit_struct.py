import numpy as np
import h5py


class DigitStruct:
    
    def __init__(self, fpath):
        self.inf = h5py.File(fpath, "r")
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']
    
    def __len__(self):
        return len(self.digitStructName)


    def get_name(self, n):
        """Return the name of the n(th) digit struct"""
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    def get_attribute(self, attr):
        """Helper function for dealing with one vs. multiple bounding boxes"""
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr
    
    def __getitem__(self, n):
        if isinstance(n, int):
            if n < 0:
                n = len(self) + n
            bbox = {}
            bb = self.digitStructBbox[n].item()
            bbox['height'] = np.array(self.get_attribute(self.inf[bb]["height"]))
            bbox['label'] = np.array(self.get_attribute(self.inf[bb]["label"]))
            bbox['left'] = np.array(self.get_attribute(self.inf[bb]["left"]))
            bbox['top'] = np.array(self.get_attribute(self.inf[bb]["top"]))
            bbox['width'] = np.array(self.get_attribute(self.inf[bb]["width"]))
            bbox['name'] = self.get_name(n)
            return bbox
        if isinstance(n, slice):
            res = []
            start = n.start if n.start is not None else 0
            if start < 0:
                start = len(self) + start
            stop = n.stop if n.stop is not None else len(self)
            if stop < 0:
                stop = len(self) + stop
            step = n.step if n.step is not None else 1
            if step < 0:
                startcp = start
                start = stop - 1
                stop = startcp - 1
            for i in range(start, stop, step):
                res.append(self[i])
            return res

        raise ValueError("You tried to pass invalid value to __getitem__")
    
    def __iter__(self):
        def iter_through():
            for n in range(len(self)):
                yield self[n]
        return iter(iter_through())
