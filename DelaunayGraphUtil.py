import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay#, delaunay_plot_2d
import os
#import matplotlib.pyplot as plt

from spektral.data import Graph, Dataset
from ImportUtils import import_data, import_files_list

class DelaunayGraph(Dataset):
    def __init__(self, graph_path, transforms=None, **kwargs):
        self.graph_path = graph_path
        super().__init__(transforms, **kwargs)
    
    def check_mesh_quality(self, points):
        a = np.linalg.norm(points[0, :] - points[1, :])
        b = np.linalg.norm(points[1, :] - points[2, :])
        c = np.linalg.norm(points[2, :] - points[0, :])

        s = (a + b + c) / 2

        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        edge_length_rms = np.sqrt(np.mean(np.square([a, b, c])))

        if (4 * (3 ** 0.5) * area) / (3 * np.square(np.abs(edge_length_rms))) >= 0.8:
            return True
        else:
            return False
    
    def download(self):
        self.files = import_files_list()
        def make_graph(disp_name, str_name):
            y, x = import_data([disp_name, str_name])
            #graph_name = disp_name.split("/")[10].split(".csv")[0]

            delete_indices = []

            tri = Delaunay(x)
            for index, simplex in enumerate(tri.simplices):
                p = tri.points[simplex, :]
                if self.check_mesh_quality(p):
                    continue
                else:
                    delete_indices.append(index)
            
            tri.simplices = np.delete(tri.simplices, delete_indices, axis=0)
            # delaunay_plot_2d(tri)
            # plt.savefig("Plots/Delaunay/" + graph_name + ".png")

            edges = np.concatenate((tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, [0, 2]]), axis=0)
            edges = np.unique(edges, axis=0)
            rows = np.concatenate((edges[:, 0], edges[:, 1]), axis=0)
            cols = np.concatenate((edges[:, 1], edges[:, 0]), axis=0)
            values = np.ones(rows.shape[0])
            a = sp.csr_matrix((values, (rows, cols)))
            e = np.expand_dims(a.data, axis=1)

            val_inputs = np.asarray([np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])])

            x[:, 0], x[:, 1] = np.true_divide(x[:, 0] - val_inputs[0], val_inputs[1] - val_inputs[0]), np.true_divide(x[:, 1] - val_inputs[2], val_inputs[3] - val_inputs[2])
            y[:, 0], y[:, 1] = np.true_divide(y[:, 0] - val_inputs[0], val_inputs[1] - val_inputs[0]), np.true_divide(y[:, 1] - val_inputs[2], val_inputs[3] - val_inputs[2])

            a = a.todense()

            return Graph(x=x, a=a, e=e, y=y, val_inputs=val_inputs)
        
        self.graphs = [make_graph(str(i), str(j)) for i, j in self.files]

        for i in range(len(self.graphs)):

            filename = os.path.join(self.graph_path, f"graph_{i}")
            np.savez(filename, x=self.graphs[i].x, a=self.graphs[i].a, e=self.graphs[i].e, y=self.graphs[i].y, val_inputs=self.graphs[i].val_inputs)
    
    def read(self):
        output = []

        files = os.listdir(self.graph_path)
        for f in files:
            data = np.load(os.path.join(self.graph_path, f), allow_pickle=True)
            output.append(Graph(x=data['x'], a=data['a'], e=data['e'], y=data['y'], val_inputs=data['val_inputs']))
        return output
        