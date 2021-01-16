---
date: 2021-01-06 21:00:00
description: Python+图+树+并查集
title: python应用
author: 鱼摆摆
comments: true
tags: 
 - python
photos: https://w.wallhaven.cc/full/nm/wallhaven-nmwp71.jpg
categories: 学习
---

# 深度优先搜索(DFS)模板:

```java
def dfs(): 
    if (满足特定条件）{
        // 返回结果 or 退出搜索空间
    }
    for choice in all_choices():
        record(choice)
        dfs()
        rollback(choice)
}
```



# 广度优先搜索(BFS)模板：

```python
class Solution:
    def bfs(k):
        # 使用双端队列，而不是数组。因为数组从头部删除元素的时间复杂度为 N，双端队列的底层实现其实是链表。
        queue = collections.deque([root])
        # 记录层数
        steps = 0
        # 需要返回的节点
        ans = []
        # 队列不空，生命不止！
        while queue:
            size = len(queue)
            # 遍历当前层的所有节点
            for _ in range(size):
                node = queue.popleft()
                if (step == k) ans.append(node)
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)
            # 遍历完当前层所有的节点后 steps + 1
            steps += 1
        return ans
```



# 典型建图代码：

```python
# 邻接表形式
from collections import defaultdict

graph = defaultdict(list)
for u, v, c in times:
    graph[u].append((v, c))
    
# 邻接矩阵形式
graph = [[float('inf') for _ in range(n)] for _ in range(n)]
for u, v, c in items:
    graph[u][v] = c
    graph[v][u] = c
```



# Dijkstra算法(狄克斯拉特算法)

模板：借助堆的数据结构，每次在$\mathcal{O} logn$ 时间复杂度内找到cost最小的点。(注：只适用于有向无环图，且无负权边)

步骤：

- 找出最短时间内前往的节点(对于该节点来说，没有更短路径，直接将其加入到visited中)
- 对于该节点的每个邻居，更新开销
- 重复计算，直到每个节点的开销都被更新过
- 计算最终路径

```python
import heapq

def dijkstra(graph, start, end):
    # 堆里的数据都是 (cost, i) 的二元祖，其含义是“从 start 走到 i 的距离是 cost”。
    heap = [(0, start)]
    visited = set()
    while heap:
        (cost, u) = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            return cost
        for v, c in graph[u]:
            if v in visited:
                continue
            next = cost + c
            heapq.heappush(heap, (next, v))
    return -1
```





# Floyd_warshall算法：

基于动态规划，$dist(i,j)=dist{i,k}+dist(j,k)$，时间复杂度$\mathcal{O}(n^3)$，空间复杂度$\mathcal{O}(n^2)$，$n$ 为图顶点个数。

```python
# graph 是邻接矩阵，v 是顶点个数
def floyd_warshall(graph, v):
    dist = [[float("inf") for _ in range(v)] for _ in range(v)]

    for i in range(v):
        for j in range(v):
            dist[i][j] = graph[i][j]

    # check vertex k against all other vertices (i, j)
    for k in range(v):
        # looping through rows of graph array
        for i in range(v):
            # looping through columns of graph array
            for j in range(v):
                if (
                    dist[i][k] != float("inf")
                    and dist[k][j] != float("inf")
                    and dist[i][k] + dist[k][j] < dist[i][j]
                ):
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist, v
```



# A星寻路算法

解决的问题是在二维表格中找出任意两点的最短路径，一般题目中包括障碍物。

$$f(n) = g(n) + h(n)$$

- $f(n)$ 是从初始状态经由状态 $n$ 到目标状态的估计代价，
- $g(n)$ 是在状态空间中从初始状态到状态 $n$ 的实际代价，
- $h(n)$ 是从状态 $n$ 到目标状态的最佳路径的估计代价。

估价算法：一般使用曼哈顿距离进行估价。

$$H(n) = D * (abs ( n.x – goal.x ) + abs ( n.y – goal.y ) )$$

```python
grid = [
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],  # 0 are free path whereas 1's are obstacles
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
]

"""
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]"""

init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]  # all coordinates are given in format [y,x]
cost = 1

# the cost map which pushes the path closer to the goal
heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
for i in range(len(grid)):
    for j in range(len(grid[0])):
        heuristic[i][j] = abs(i - goal[0]) + abs(j - goal[1])
        if grid[i][j] == 1:
            heuristic[i][j] = 99  # added extra penalty in the heuristic map


# the actions we can take
delta = [[-1, 0], [0, -1], [1, 0], [0, 1]]  # go up  # go left  # go down  # go right


# function to search the path
def search(grid, init, goal, cost, heuristic):

    closed = [
        [0 for col in range(len(grid[0]))] for row in range(len(grid))
    ]  # the reference grid
    closed[init[0]][init[1]] = 1
    action = [
        [0 for col in range(len(grid[0]))] for row in range(len(grid))
    ]  # the action grid

    x = init[0]
    y = init[1]
    g = 0
    f = g + heuristic[init[0]][init[0]]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(cell) == 0:
            return "FAIL"
        else:  # to choose the least costliest action so as to move closer to the goal
            cell.sort()
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]

            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i
    invpath = []
    x = goal[0]
    y = goal[1]
    invpath.append([x, y])  # we get the reverse path from here
    while x != init[0] or y != init[1]:
        x2 = x - delta[action[x][y]][0]
        y2 = y - delta[action[x][y]][1]
        x = x2
        y = y2
        invpath.append([x, y])

    path = []
    for i in range(len(invpath)):
        path.append(invpath[len(invpath) - 1 - i])
    print("ACTION MAP")
    for i in range(len(action)):
        print(action[i])

    return path


a = search(grid, init, goal, cost, heuristic)
for i in range(len(a)):
    print(a[i])
```





# 拓扑排序：

对于有向无环图进行排序，使得对于从顶点$u$ 到顶点$v$ 的每个有向边$uv$，$u$ 在排序中都在$v$ 之前。



## Kahn算法：

假设 $L$ 是存放结果的列表，先找到那些入度为零的节点，把这些节点放到 $L$ 中，因为这些节点没有任何的父节点。然后把与这些节点相连的边从图中去掉，再寻找图中的入度为零的节点。对于新找到的这些入度为零的节点来说，他们的父节点已经都在$L$ 中了，所以也可以放入 $L$。重复上述操作，直到找不到入度为零的节点。如果此时 $L$ 中的元素个数和节点总数相同，说明排序完成；如果$L$ 中的元素个数和节点总数不同，说明原图中存在环，无法进行拓扑排序。

```python
def topologicalSort(graph):
    """
    Kahn's Algorithm is used to find Topological ordering of Directed Acyclic Graph
    using BFS
    """
    indegree = [0] * len(graph)
    queue = []
    topo = []
    cnt = 0

    for key, values in graph.items():
        for i in values:
            indegree[i] += 1

    for i in range(len(indegree)):
        if indegree[i] == 0:
            queue.append(i)

    while queue:
        vertex = queue.pop(0)
        cnt += 1
        topo.append(vertex)
        for x in graph[vertex]:
            indegree[x] -= 1
            if indegree[x] == 0:
                queue.append(x)

    if cnt != len(graph):
        print("Cycle exists")
    else:
        print(topo)


# Adjacency List of Graph
graph = {0: [1, 2], 1: [3], 2: [3], 3: [4, 5], 4: [], 5: []}
topologicalSort(graph)
```



## 最小生成树：

## Krusal算法：

```python
from typing import List, Tuple


def kruskal(num_nodes: int, num_edges: int, edges: List[Tuple[int, int, int]]) -> int:
    """
    >>> kruskal(4, 3, [(0, 1, 3), (1, 2, 5), (2, 3, 1)])
    [(2, 3, 1), (0, 1, 3), (1, 2, 5)]

    >>> kruskal(4, 5, [(0, 1, 3), (1, 2, 5), (2, 3, 1), (0, 2, 1), (0, 3, 2)])
    [(2, 3, 1), (0, 2, 1), (0, 1, 3)]

    >>> kruskal(4, 6, [(0, 1, 3), (1, 2, 5), (2, 3, 1), (0, 2, 1), (0, 3, 2),
    ... (2, 1, 1)])
    [(2, 3, 1), (0, 2, 1), (2, 1, 1)]
    """
    edges = sorted(edges, key=lambda edge: edge[2])

    parent = list(range(num_nodes))

    def find_parent(i):
        if i != parent[i]:
            parent[i] = find_parent(parent[i])
        return parent[i]

    minimum_spanning_tree_cost = 0
    minimum_spanning_tree = []

    for edge in edges:
        parent_a = find_parent(edge[0])
        parent_b = find_parent(edge[1])
        if parent_a != parent_b:
            minimum_spanning_tree_cost += edge[2]
            minimum_spanning_tree.append(edge)
            parent[parent_a] = parent_b

    return minimum_spanning_tree


if __name__ == "__main__":  # pragma: no cover
    num_nodes, num_edges = list(map(int, input().strip().split()))
    edges = []

    for _ in range(num_edges):
        node1, node2, cost = [int(x) for x in input().strip().split()]
        edges.append((node1, node2, cost))

    kruskal(num_nodes, num_edges, edges)
```



## Prim算法：

```python
import sys
from collections import defaultdict


def PrimsAlgorithm(l):  # noqa: E741

    nodePosition = []

    def get_position(vertex):
        return nodePosition[vertex]

    def set_position(vertex, pos):
        nodePosition[vertex] = pos

    def top_to_bottom(heap, start, size, positions):
        if start > size // 2 - 1:
            return
        else:
            if 2 * start + 2 >= size:
                m = 2 * start + 1
            else:
                if heap[2 * start + 1] < heap[2 * start + 2]:
                    m = 2 * start + 1
                else:
                    m = 2 * start + 2
            if heap[m] < heap[start]:
                temp, temp1 = heap[m], positions[m]
                heap[m], positions[m] = heap[start], positions[start]
                heap[start], positions[start] = temp, temp1

                temp = get_position(positions[m])
                set_position(positions[m], get_position(positions[start]))
                set_position(positions[start], temp)

                top_to_bottom(heap, m, size, positions)

    # Update function if value of any node in min-heap decreases
    def bottom_to_top(val, index, heap, position):
        temp = position[index]

        while index != 0:
            if index % 2 == 0:
                parent = int((index - 2) / 2)
            else:
                parent = int((index - 1) / 2)

            if val < heap[parent]:
                heap[index] = heap[parent]
                position[index] = position[parent]
                set_position(position[parent], index)
            else:
                heap[index] = val
                position[index] = temp
                set_position(temp, index)
                break
            index = parent
        else:
            heap[0] = val
            position[0] = temp
            set_position(temp, 0)

    def heapify(heap, positions):
        start = len(heap) // 2 - 1
        for i in range(start, -1, -1):
            top_to_bottom(heap, i, len(heap), positions)

    def deleteMinimum(heap, positions):
        temp = positions[0]
        heap[0] = sys.maxsize
        top_to_bottom(heap, 0, len(heap), positions)
        return temp

    visited = [0 for i in range(len(l))]
    Nbr_TV = [-1 for i in range(len(l))]  # Neighboring Tree Vertex of selected vertex
    # Minimum Distance of explored vertex with neighboring vertex of partial tree
    # formed in graph
    Distance_TV = []  # Heap of Distance of vertices from their neighboring vertex
    Positions = []

    for x in range(len(l)):
        p = sys.maxsize
        Distance_TV.append(p)
        Positions.append(x)
        nodePosition.append(x)

    TreeEdges = []
    visited[0] = 1
    Distance_TV[0] = sys.maxsize
    for x in l[0]:
        Nbr_TV[x[0]] = 0
        Distance_TV[x[0]] = x[1]
    heapify(Distance_TV, Positions)

    for i in range(1, len(l)):
        vertex = deleteMinimum(Distance_TV, Positions)
        if visited[vertex] == 0:
            TreeEdges.append((Nbr_TV[vertex], vertex))
            visited[vertex] = 1
            for v in l[vertex]:
                if visited[v[0]] == 0 and v[1] < Distance_TV[get_position(v[0])]:
                    Distance_TV[get_position(v[0])] = v[1]
                    bottom_to_top(v[1], get_position(v[0]), Distance_TV, Positions)
                    Nbr_TV[v[0]] = vertex
    return TreeEdges


if __name__ == "__main__":  # pragma: no cover
    # < --------- Prims Algorithm --------- >
    n = int(input("Enter number of vertices: ").strip())
    e = int(input("Enter number of edges: ").strip())
    adjlist = defaultdict(list)
    for x in range(e):
        l = [int(x) for x in input().strip().split()]  # noqa: E741
        adjlist[l[0]].append([l[1], l[2]])
        adjlist[l[1]].append([l[0], l[2]])
    print(PrimsAlgorithm(adjlist))
```



# 二分图：

参考leetcode 886，785

代码模板：

```python
from typing import List
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        '''
        :param graph: 输入邻接表方式给出的图，例如：
        [[1,3], [0,2], [1,3], [0,2]]
        图如下：
        0----1
        |    |
        |    |
        3----2
        :return: True or False
        '''
        n = len(graph)
        colors = [0 for _ in range(n)]
        def dfs(i, color):
            colors[i] = color
            for j in graph[i]:
                if colors[j] == color:
                    return False
                if colors[j] == 0 and not dfs(j, -1 * color):
                    return False
            return True
        for i in range(n):
            if colors[i] == 0 and not dfs(i, 1):
                return False
        return True
```



# 并查集：

```python
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}
    
    def find(self,x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] != None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
         
        return root
    
    def merge(self,x,y,val):
        """
        合并两个节点
        """
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self,x,y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)
    
    def add(self,x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None
```



```python
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        记录每个节点到根节点的权重
        """
        self.father = {}
        self.value = {}
    
    def find(self,x):
        """
        查找根节点
        路径压缩
        更新权重
        """
        root = x
        # 节点更新权重的时候要放大的倍数
        base = 1
        while self.father[root] != None:
            root = self.father[root]
            base *= self.value[root]
        
        while x != root:
            original_father = self.father[x]
            ##### 离根节点越远，放大的倍数越高
            self.value[x] *= base
            base /= self.value[original_father]
            #####
            self.father[x] = root
            x = original_father
         
        return root
```



```python
from collections import defaultdict
class DisjointSet:
    def __init__(self):
        # self.father 存储父节点
        # self.rank 存储树深
        self.father = {}
        self.rank = defaultdict(int)

    def find(self, x):
        # 若x不存在父节点，将其父节点置为自身
        self.father.setdefault(x, x)
        # 循环或迭代查找父节点
        ##########################
        # while x != self.father[x]:
        #     x = self.father[x]
        #########################
        # 迭代过程中采用路径压缩
        while x != self.father[x]:
            self.father[x] = self.find(self.father[x])
            x = self.father[x]
        return x

    def union(self, A, B):
        a, b = self.find(A), self.find(B)
        # 父节点合并，调整深度
        if a == b:
            return
        elif self.rank[a] <= self.rank[b]:
            self.father[a] = b
            if self.rank[a] == self.rank[b]:
                self.rank[b] += 1
        else:
            self.father[b] = a

    def same(self, x, y):
        return self.find(x) == self.find(y)

    # 给定节点列表，建图
    def build(self, nodes):
        for u, v in nodes:
            self.union(u, v)
```

