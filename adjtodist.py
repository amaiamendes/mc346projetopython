from collections import deque, namedtuple
import numpy as np
import sys


# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
    return Edge(start, end, cost)

class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            # this piece of magic turns ([1,2], [3,4]) into [1, 2, 3, 4]
            # the set above makes it's elements unique.
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'

        # 1. Mark all nodes unvisited and store them.
        # 2. Set the distance to zero for our initial node
        # and to infinity for other nodes.
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            # 3. Select the unvisited node with the smallest distance,
            # it's current node now.
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])

            # 6. Stop, if the smallest distance
            # among the unvisited nodes is infinity.
            if distances[current_vertex] == inf:
                break

            # 4. Find unvisited neighbors for the current node
            # and calculate their distances through the current node.
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost

                # Compare the newly calculated distance to the assigned
                # and save the smaller one.
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

            # 5. Mark the current node as visited
            # and remove it from the unvisited set.
            vertices.remove(current_vertex)

        # print(distances)
        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return distances, path

content = []
for line in sys.stdin:
    line = line.strip()
    content.append(line)

for i in range(0,len(content)):
    # Removido a verificação isDigit e o cast para int. Motivo: no teste3 tem entradas do tipo float.
    content[i] = [float(s) for s in content[i].split()]


g = True
glist = []
p = []
 # Indíce para identificar o passageiro na saída final
passengerIndex = 0
for x in content:
    if x == []:
        g = False
        continue
    if g == True:
        print('vertice:', x)
        glist.append((x[0], x[1], x[2]))
    else:
        if len(x) == 2:
            p.append((passengerIndex, x[0], x[1], -1))
        else:
            p.append((passengerIndex, x[0], x[1], x[2]))
        # Atualiza o indíce para o próximo passageiro
        passengerIndex = passengerIndex + 1

print(content)
print(glist)
print('\nPassageiros')
print(p)

graph = Graph(glist)

dist = {}
for v in graph.vertices:
	dist[v], o = graph.dijkstra(v,v)

print('\nDistancia entre os vertices')
for x in dist:
    print (x)
    for y in dist[x]:
        print (y,':',dist[x][y])


class pool():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def best_pool(self, s1, s2, d1, d2, x1, x2, dist):
        self.pool_path = ()
        self.pool_incov_max_min = np.inf

        if x1 != -1:
            s1 = x1
        if x2 != -1:
            s2 = x2

        if x2 == -1:
        # s1, s2, d2, d1
            incov1 = (dist[s1][s2]+dist[s2][d2]+dist[d2][d1])/dist[s1][d1]
            incov2 = dist[s2][d2]/dist[s2][d2]
            path = (s1, s2, d2, d1)
            if max(incov1,incov2) <= 1.4:
                incov_max = max(incov1,incov2)
            # print("path:", path)
            # print("incov1=", incov1)
            # print("incov2=", incov2)
            # print("incov_max=", incov_max)

                if incov_max < self.pool_incov_max_min:
                    self.pool_path = path
                    self.pool_incov_max_min = incov_max
            # print("pool_path:", self.pool_path)
            # print("pool_incov_max_min:", self.pool_incov_max_min, "\n")

            # s1, s2, d1, d2
            incov1 = (dist[s1][s2]+dist[s2][d1])/dist[s1][d1]
            incov2 = (dist[s2][d1]+dist[d1][d2])/dist[s2][d2]
            path = (s1, s2, d1, d2)
            if max(incov1,incov2) <= 1.4:
                incov_max = max(incov1,incov2)
            # print("path:", path)
            # print("incov1=", incov1)
            # print("incov2=", incov2)
            # print("incov_max=", incov_max)

                if incov_max < self.pool_incov_max_min:
                    self.pool_path = path
                    self.pool_incov_max_min = incov_max
            # print("pool_path:", self.pool_path)
            # print("pool_incov_max_min:", self.pool_incov_max_min, "\n")

        if x1 == -1:

        # s2, s1, d2, d1
            incov1 = (dist[s1][d2]+dist[d2][d1])/dist[s1][d1]
            incov2 = (dist[s2][s1]+dist[s1][d2])/dist[s2][d2]
            path = (s2, s1, d2, d1)
            if max(incov1,incov2) <= 1.4:
                incov_max = max(incov1,incov2)
            # print("path:", path)
            # print("incov1=", incov1)
            # print("incov2=", incov2)
            # print("incov_max=", incov_max)

                if incov_max < self.pool_incov_max_min:
                    self.pool_path = path
                    self.pool_incov_max_min = incov_max
            # print("pool_path:", self.pool_path)
            # print("pool_incov_max_min:", self.pool_incov_max_min, "\n")

            # s2, s1, d1, d2
            incov1 = dist[s1][d1]/dist[s1][d1]
            incov2 = (dist[s2][s1]+dist[s1][d1]+dist[d1][d2])/dist[s2][d2]
            path = (s2, s1, d1, d2)
            if max(incov1,incov2) <= 1.4:
                incov_max = max(incov1,incov2)
            # print("path:", path)
            # print("incov1=", incov1)
            # print("incov2=", incov2)
            # print("incov_max=", incov_max)

                if incov_max < self.pool_incov_max_min:
                    self.pool_path = path
                    self.pool_incov_max_min = incov_max
            # print("pool_path:", self.pool_path)
            # print("pool_incov_max_min:", self.pool_incov_max_min, "\n")

        return self.p1, self.p2,self.pool_path,self.pool_incov_max_min

a = {}
for i in range(0,len(p)):
    a[i] = {}
    for j in range(0,len(p)):
        if i != j:
            # Desloquei as entradas para best_pool um elemento para a direita por causa da inserção do índice do passageiro
            a[i][j] = pool(p[i],p[j]).best_pool(p[i][1], p[j][1], p[i][2], p[j][2], p[i][3], p[j][3], dist)

for x in a:
    print (x)
    for y in a[x]:
        print (y,':',a[x][y])

unavailable = []
pools = []
for x in a:
    print(unavailable)
    print(x)
    if x not in unavailable:
        min_incov = np.inf
        min_pool = -1
        for y in a[x]:
            if y not in unavailable:
                if a[x][y][3] < min_incov:
                    min_incov = a[x][y][3]
                    min_pool = y
        if min_incov != np.inf:
            pools.append(a[x][min_pool])
            unavailable.append(x)
            unavailable.append(min_pool)
        else:
            pools.append(p[3])

print("\npools")
makeRoute = lambda route: ' '.join(map(str, [int(s) for s in route]))
for x in pools:
    # Se for uma tupla de passageiros no pool, então o parse é diferente.
    if(type(x[0]) == tuple):
        print('passageiros:', x[0][0], x[1][0], 'percurso:', makeRoute(x[2]))
    else:
        print('passageiro:', x[0], 'percurso:', int(x[1]), int(x[2]))






# print(graph.vertices)
# print(graph.dijkstra(0, 4))
