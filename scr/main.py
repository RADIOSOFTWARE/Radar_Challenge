# -----------import-library----------------
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import csv
import math
from numba import cuda

"""
>[!Note]на *** артикли!
##to-do
[x] unpacking
[x] packaging
~~~[ ] +- brutforse on GPU~~~
[ ] algorithm
    - [x] preparing data for graph
    - [x] preparing inverted data for graph
    - [x] painting graph
    - [ ] painting weights on graph
    - [ ] seatching way
    - [ ] build data for packaging

## notes
try gpu computing
"""


def create_graph(matrix: np.ndarray) -> nx.Graph:
    """
    Creating networkx graph dased on a compatibility matrix
    matrix: np.ndarray - input compatibility matrix
    return - > networkx graph
    """
    points = []
    for row in range(len(matrix)):
        for com in range(row + 1, len(matrix)):
            # autopep8: off
            if matrix[row, com]: continue
            # autopip8: on
            points.append((row, com)) 
    graph = nx.Graph()
    graph.add_edges_from(points)
    return graph


def create_graph_testing(matrix: np.ndarray) -> nx.Graph:
    """
    !old_version
    Creating networkx graph dased on a compatibility matrix
    matrix: np.ndarray - input compatibility matrix
    return - > networkx graph
    """
    rows, cols = np.where(np.invert(matrix))
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph

def painting_graph(graph: nx.Graph) -> None:
    """
    drawning graph using matplotlib
    graph: nx.Graph - networkx graph
    """
    nx.draw(graph, node_size = 250, with_labels=True)
    plt.show() 
    pass


def algorithm(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    method of seatching for global hypotheses
    weights: np.ndarray - input weights of rout hypotheses
    matrix: np.ndarray - input compatibility matrix
    """
    
    graph = create_graph(matrix)
    painting_graph(graph)

    return matrix

def enum (matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
     
    """ 
    GPU computing
    """
    n = matrix.shape[1]
    matrix_enum = np.zeros([n*n,n], dtype=np.float32)
    matrix_weights = np.zeros(n*n, dtype=np.float32)
    matrix_out = np.zeros([n*n,n+1], dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for z in range(n):
                matrix_enum[i*n+j,z] = matrix[i,j] * matrix[i,z]
    
    for i in range(n*n):
        if np.array_equal(matrix_enum[i-1], matrix_enum[i]):
            continue
        else:
            for j in range(n): 
                matrix_weights[i] = matrix_weights[i] + matrix_enum[i,j] * weights[j]
            matrix_out[i, 0:n-1] = matrix_enum[i, 0:n-1]
            matrix_out[i,n] = matrix_weights[i]
        matrix_sorted = matrix_out[matrix_out[:, n]. argsort ()[::-1]]
        print("matrix_weights", matrix_weights)
    return matrix_sorted

def packaging(patch_file: str, data: np.ndarray) -> None:
    """
    method of packing of global hypotheses into a csv table
    """
    quantity_column = len(data[0])
    with open(patch_file, "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        writer.writerow(list(range(quantity_column)) + ["W"])
        writer.writerows(data)


def main():
    # ---------------parser-config------------------
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--i", help="Input compatibility matrix file in csv format")
    arg_parser.add_argument(
        "--o", help="Output global hypotheses file in csv", default="output.csv"
    )

    # ---------------parsing-parameters-------------
    args = arg_parser.parse_args()
    input_file = args.i
    output_file = args.o
    
    # ---------------parsing-file-------------------
    matrix: np.ndarray
    weights: np.ndarray
    with open(input_file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t",
                                quoting=csv.QUOTE_NONNUMERIC)
        data = np.array(list(csv_reader), dtype=np.float32)
        weights = data[:, -1:]
        matrix = data[:, :-1] > 0
        weights = np.rot90(weights)[0]
    # print(matrix, weights)

    # ------------------start-----------------------
    output_data = enum(matrix, weights)
    print("output_data=",output_data)
    print("shape",output_data.shape)
    packaging(output_file, output_data)
    

if __name__ == "__main__":
    main()
