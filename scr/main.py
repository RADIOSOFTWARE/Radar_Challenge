# -----------import-library----------------
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import csv

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
        -[x] saving image compatibility graph
        -[x] saving image incompatibility graph
    - [ ] adding weights on graph
    - [x] 10:10 check point
    - [ ] seatching way
    - [ ] build data for packaging
    - [ ] Решение задач на оптизацию/ линейно епрограмирование
## Tehnologies:
- python
- numpy
- networkx
## notes
try gpu computing
"""


def create_graph_incompatibility(matrix: np.ndarray) -> nx.Graph:
    """
    Creating networkx graph dased on compatibility matrix
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


def create_graph_campatibility(matrix: np.ndarray) -> nx.Graph:
    """
    !old_version
    Creating networkx graph dased on compatibility matrix
    matrix: np.ndarray - input compatibility matrix
    return - > networkx graph
    """
    rows, cols = np.where(matrix)
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
    graph = create_graph_incompatibility(matrix)
    painting_graph(graph)

    return matrix


def packaging(patch_file: str, data: np.ndarray) -> None:
    """
    method of packing of global hypotheses into csv table
    patch_file: str - path to file where output will be located if there is no output, create new one
    data: np.ndarray - array of global hypotheses
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
        weights = np.rot90(weights)
    # print(matrix, weights)

    # ------------------start-----------------------
    output_data: np.ndarray = algorithm(matrix, weights)
    packaging(output_file, output_data)


if __name__ == "__main__":
    main()
