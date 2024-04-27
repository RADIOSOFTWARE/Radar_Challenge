# -----------import-library----------------
import matplotlib
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
    - [ ] preparing data for graph
    - [ ] preparing inverted data for graph
    - [ ] painting graph
    - [ ] painting weights on graph
    - [ ] seatching way
    - [ ] build data for packaging

## notes
try gpu computing
"""

# -------------oldcode--------------------------------


def preparing_matrix(matrix: np.ndarray) -> list:
    points = []
    for row in range(len(matrix) - 1):
        for com in range(row + 1, len(matrix) - 1):
            # autopep8: off
            if matrix[row, com]: continue
            # autopip8: on
            points.append((row, com)) 

    return points

def create_graph(matrix: np.ndarray) -> nx.Graph:
    """
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
    nx.draw(graph, node_size = 100)
    plt.show() 
    pass


def algorithm(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    method of seatching for global hypotheses
    weights: np.ndarray - input weights of rout hypotheses
    matrix: np.ndarray - input compatibility matrix
    """
    graph: nx.Graph = create_graph(matrix)
    painting_graph(graph)

    return matrix


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
        weights = np.rot90(weights)
    # print(matrix, weights)

    # ------------------start-----------------------
    output_data: np.ndarray = algorithm(matrix, weights)
    packaging(output_file, output_data)


if __name__ == "__main__":
    main()
