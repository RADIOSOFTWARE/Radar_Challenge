# -----------import-library----------------
import time
# datatype
import numpy as np
# graph
import networkx as nx
import matplotlib.pyplot as plt
# parse-data
import argparse
import csv
# linear-programing
from scipy.optimize import linprog
import pulp as pl
"""
>[!Note]на *** артикли!
##to-do
[x] unpacking
[x] packaging
~~~[?] +- brutforse on GPU~~~
[ ] algorithm
    - [x] preparing data for graph
    - [x] preparing inverted data for graph
    - [x] painting graph
        -[x] saving image compatibility graph
        -[x] saving image incompatibility graph
    - [x] adding weights on graph
    - [x] 10:10 check point
    - [x] 18:10 second check point
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


def create_graph_incompatibility(matrix: np.ndarray, weights: np.ndarray) -> nx.Graph:
    """
    Creating networkx graph dased on compatibility matrix
    matrix: np.ndarray - input compatibility matrix
    return - > networkx graph
    """
    graph = nx.Graph()
    for row in range(len(matrix)):
        for com in range(row + 1, len(matrix)):
            # autopep8: off
            if matrix[row, com]: continue
            # autopip8: on
            graph.add_edge(row, com, weight = weights[row] + weights[com]) 

    return graph



def painting_graph(graph: nx.Graph) -> None:
    """
    drawning graph using matplotlib
    graph: nx.Graph - networkx graph
    """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size = 250, with_labels=True)
    # nx.draw_networkx_edge_labels(graph,pos)
    plt.show()


def linear_algorithm(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = matrix + np.transpose(matrix)
    # print(maматрицы смежности
    matrix = matrix*1

    weights_title = np.tile(weights, (len(weights), 1))
    print(np.sum(matrix * weights, axis = 1))
    result = linprog(c=-weights, A_ub=matrix*weights,
                     b_ub=np.sum(matrix * weights_title, axis = 1), 
                     bounds=[(0, 1)]*len(matrix))


    # Вывод результата
    print("Оптимальное значение:", result.fun)
    for i, res in enumerate(result.x):
        if res == 0:
            continue
        print(f"point - {i}\nweight - {weights[i]}")
    print("Оптимальное решение:", result.x)





def algorithm(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    method of seatching for global hypotheses
    weights: np.ndarray - input weights of rout hypotheses
    matrix: np.ndarray - input compatibility matrix
    """
    graph = create_graph_incompatibility(matrix, weights)
    # linear_algorithm(matrix, weights)
    nodes_with_one_edge = [node for node, degree in graph.degree() if degree == 1]
    sorted_nodes = sorted(graph.degree(weight="weight"), key=lambda x: x[1], reverse=False)
    points = []
    neighbors = []
    print(len(sorted_nodes))
    for i in range(len(sorted_nodes)):
        if i == len(sorted_nodes) - 1:
            break
        point = sorted_nodes[i][0]
        # print(neighbors)
        if point in neighbors:
            del sorted_nodes[i]
            continue
        points.append(point)
        neighbors+=list(graph.neighbors(point))

    global_hypothesis = np.ones(len(matrix))
    global_hypothesis[points] = 0
    # w = sum(weights) - sum(weights[path])
    print(points)
    w = sum(global_hypothesis*weights)
    print(list(global_hypothesis.astype(int)))
    print(w)






    print(len(sorted_nodes))
    print(nodes_with_one_edge)
    paths = []

    for i in range(len(nodes_with_one_edge)-1):
        print("-")
        path = nx.astar_path(graph, nodes_with_one_edge[i], nodes_with_one_edge[i+1], weight='weight')
        paths.append(path)
        points = []
        for point in graph.edges(path):
            points.append(point[-1])
        points.append(path[-1])
        print(path)
        print(points)
        global_hypothesis = np.ones(len(matrix))
        global_hypothesis[points] = 0
        # w = sum(weights) - sum(weights[path])
        w = sum(global_hypothesis*weights)
        print(list(global_hypothesis.astype(int)))
        print(list(range(len(matrix))))
    #
    painting_graph(graph)


    # print(np.transpose(matrix)*1)
    return matrix * weights


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
        csv_reader = csv.reader(csv_file, delimiter=",",
                                quoting=csv.QUOTE_NONNUMERIC)
        data = np.array(list(csv_reader), dtype=np.float32)
        # print(data)
        weights = data[-1]
        matrix = data[0:-1] > 0
        # print(matrix, weights)

    # ------------------start-----------------------
    output_data: np.ndarray = algorithm(matrix, weights)
    packaging(output_file, output_data)


if __name__ == "__main__":
    main()
