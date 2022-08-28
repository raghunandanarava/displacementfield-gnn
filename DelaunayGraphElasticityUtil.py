import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
import os
import matplotlib.pyplot as plt
import pdb

from spektral.data import Graph, Dataset
from ImportUtils2 import import_data, import_files_list

class DelaunayGraph(Dataset):
    def __init__(self, graph_path, transforms=None, **kwargs):
        """
        mins = [ 5.19950000e+01  1.99950000e+01 -9.07300000e-01 -1.47200000e+01 -1.56071271e-02 -3.17241379e-01 -7.00459388e-02 -1.44518289e-01]
        maxs = [86.005      55.205      31.34       17.47        0.59576923  0.31648551 0.51731128  0.45680483]
        """
        self.graph_path = graph_path
        self.x_max = 86.005
        self.x_min = 51.995
        self.y_max = 55.205
        self.y_min = 19.995
        self.ux_max = 31.34
        self.ux_min = -0.9073
        self.uy_max = 17.47
        self.uy_min = -14.72
        self.strainx_max = 0.59576923
        self.strainx_min = -0.015607127
        self.strainy_max = 0.31648551
        self.strainy_min = -0.317241379
        self.stressx_max = 0.51731128
        self.stressx_min = -0.0700459388
        self.stressy_max = 0.45680483
        self.stressy_min = -0.144518289
        self.max_pitch = 0.06
        self.min_pitch = 0.028
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
        def make_graph(disp_name):
            y, x_initial = import_data(disp_name)
            #graph_name = disp_name.split("/")[10].split(".csv")[0]

            delete_indices = []

            tri = Delaunay(x_initial[:, 0:2])
            # delaunay_plot_2d(tri)
            # plt.savefig("Plots/Delaunay/" + disp_name + ".png")
            for index, simplex in enumerate(tri.simplices):
                p = tri.points[simplex, :]
                if self.check_mesh_quality(p):
                    continue
                else:
                    delete_indices.append(index)
            
            tri.simplices = np.delete(tri.simplices, delete_indices, axis=0)
            # delaunay_plot_2d(tri)
            # plt.savefig("Plots/Delaunay/" + disp_name + "____.png")

            edges = np.concatenate((tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, [0, 2]]), axis=0)
            edges = np.unique(edges, axis=0)
            rows = np.concatenate((edges[:, 0], edges[:, 1]), axis=0)
            cols = np.concatenate((edges[:, 1], edges[:, 0]), axis=0)
            values = np.ones(rows.shape[0])
            a = sp.csr_matrix((values, (rows, cols)))
            e = np.expand_dims(a.data, axis=1)

            # x_initial[:, 0], x_initial[:, 1] = np.divide(
            #     np.subtract(x_initial[:, 0], self.x_min), np.subtract(self.x_max, self.x_min)
            # ), np.divide(
            #     np.subtract(x_initial[:, 1], self.y_min), np.subtract(self.y_max, self.y_min)
            # )

            y[:, 0], y[:, 1] = np.divide(
                np.subtract(y[:, 0], self.ux_min), np.subtract(self.ux_max, self.ux_min)
            ), np.divide(
                np.subtract(y[:, 1], self.uy_min), np.subtract(self.uy_max, self.uy_min)
            )

            y[:, 2], y[:, 3] = np.divide(
                np.subtract(y[:, 2], self.strainx_min), np.subtract(self.strainx_max, self.strainx_min)
            ), np.divide(
                np.subtract(y[:, 3], self.strainy_min), np.subtract(self.strainy_max, self.strainy_min)
            )

            y[:, 4], y[:, 5] = np.divide(
                np.subtract(y[:, 4], self.stressx_min), np.subtract(self.stressx_max, self.stressx_min)
            ), np.divide(
                np.subtract(y[:, 5], self.stressy_min), np.subtract(self.stressy_max, self.stressy_min)
            )
            #y = np.column_stack((y, u_middle, stra, stre))

            val_inputs = np.asarray([np.min(x_initial[:, 0]), np.max(x_initial[:, 0]), np.min(x_initial[:, 1]), np.max(x_initial[:, 1])])

            x_initial[:, 0], x_initial[:, 1] = np.true_divide(x_initial[:, 0] - val_inputs[0], val_inputs[1] - val_inputs[0]), np.true_divide(x_initial[:, 1] - val_inputs[2], val_inputs[3] - val_inputs[2])
            #x_initial[:, 2] = np.true_divide(x_initial[:, 2] - self.min_pitch, self.max_pitch - self.min_pitch)
            #y[:, 0], y[:, 1] = np.true_divide(y[:, 0] - val_inputs[0], val_inputs[1] - val_inputs[0]), np.true_divide(y[:, 1] - val_inputs[2], val_inputs[3] - val_inputs[2])

            a = a.todense()

            return Graph(x=x_initial, a=a, e=e, y=y, val_inputs=val_inputs)
        
        self.graphs = [make_graph(str(i)) for i in self.files]

        for i in range(len(self.graphs)):

            filename = os.path.join(self.graph_path, f"graph_{i}")
            #np.savez(filename, x=self.graphs[i].x, a=self.graphs[i].a, e=self.graphs[i].e, y=self.graphs[i].y)
            np.savez(filename, x=self.graphs[i].x, a=self.graphs[i].a, e=self.graphs[i].e, y=self.graphs[i].y, val_inputs=self.graphs[i].val_inputs)
    
    def read(self):
        output = []

        files = os.listdir(self.graph_path)
        for f in files:
            data = np.load(os.path.join(self.graph_path, f), allow_pickle=True)
            #output.append(Graph(x=data['x'], a=data['a'], e=data['e'], y=data['y']))
            output.append(Graph(x=data['x'], a=data['a'], e=data['e'], y=data['y'], val_inputs=data['val_inputs']))
        return output
        #return self.graphs
  