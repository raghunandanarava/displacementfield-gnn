import numpy as np
from sklearn.neighbors import kneighbors_graph
from ImportUtils import *
import pdb

from spektral.data import Dataset, Graph

from DataDiscretisationUtil import vrange


class GraphData(Dataset):
    def __init__(self, transforms, **kwargs):
        self.files = import_files_list()
        self.graph_path = '/home/data/litho/Raghu/Graphs/NotUsed'
        super().__init__(transforms=transforms, **kwargs)
    
    def download(self):
        self.graphs = [self.make_graph(str(i), str(j)) for i, j in self.files]

        for i in range(len(self.graphs)):
            filename = os.path.join(self.graph_path, f'graph_{i}')

            np.savez(filename, x=self.graphs[i].x, a=self.graphs[i].a, e=self.graphs[i].e ,y=self.graphs[i].y, val_inputs=self.graphs[i].val_inputs)
    
    def make_graph(self, disp_name, str_name):
        y, x = import_data([disp_name, str_name])

        # discretised_x = vrange(x[:, 0], y[:, 0], 5)
        # discretised_y = vrange(x[:, 1], y[:, 1], 5)

        x_min_input, x_max_input, y_min_input, y_max_input = np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])

        #x_min_output, x_max_output, y_min_output, y_max_output = np.min(y[:, 0]), np.max(y[:, 0]), np.min(y[:, 1]), np.max(y[:, 1])

        x[:, 0], x[:, 1] = np.true_divide(x[:, 0] - x_min_input, x_max_input - x_min_input), np.true_divide(x[:, 1] - y_min_input, y_max_input - y_min_input)
        y[:, 0], y[:, 1] = np.true_divide(y[:, 0] - x_min_input, x_max_input - x_min_input), np.true_divide(y[:, 1] - y_min_input, y_max_input - y_min_input)

        # discretised_x, discretised_y = np.true_divide(discretised_x - x_min, x_max - x_min), np.true_divide(discretised_y - y_min, y_max -y_min)

        val_inputs = np.asarray([x_min_input, x_max_input, y_min_input, y_max_input])
        #val_outputs = np.asarray([x_min_output, x_max_output, y_min_output, y_max_output])
        a = kneighbors_graph(x, 4, mode='distance', metric='euclidean', include_self=False)
        e = np.expand_dims(a.data, axis=1)
        a.data = np.ones_like(a.data)
        a = a.todense()

        return Graph(x=x, a=a, e=e, y=y, val_inputs=val_inputs)
    
    def read(self):
        output = []

        files = os.listdir(self.graph_path)
        for f in files:
            data = np.load(os.path.join(self.graph_path, f), allow_pickle=True)
            output.append(Graph(x=data['x'], a=data['a'], e=data['e'], y=data['y'], val_inputs=data['val_inputs']))
        return output
