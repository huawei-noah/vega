"""DAG class."""

from collections import deque
from collections import OrderedDict


class DAG:
    """DAG."""

    def __init__(self):
        """Init DAG."""
        self.nodes = OrderedDict()

    def add_node(self, node):
        """Add node."""
        if node not in self.nodes:
            self.nodes[node] = set()

    def remove_node(self, node):
        """Remove node."""
        if node in self.nodes:
            self.nodes.pop(node)

        for pre_node, nodes in iter(self.nodes.items()):
            if node in nodes:
                nodes.remove(node)

    def add_edge(self, pre_node, node):
        """Add edge."""
        if pre_node not in self.nodes or node not in self.nodes:
            return
        self.nodes[pre_node].add(node)

    def remove_edge(self, pre_node, node):
        """Remove edge."""
        if pre_node in self.nodes and node in self.nodes[pre_node]:
            self.nodes[pre_node].remove(node)

    def from_dict(self, dict_value):
        """Construct DAG from dict."""
        self.nodes = OrderedDict()
        for node in iter(dict_value.keys()):
            self.add_node(node)
        for pre_node, nodes in iter(dict_value.items()):
            if not isinstance(nodes, list):
                raise TypeError('dict values must be lists')
            for node in nodes:
                self.add_edge(pre_node, node)

    def next_nodes(self, node):
        """Get all successor of the node."""
        return list(self.nodes[node])

    def pre_nodes(self, node):
        """Get all predecessor of the node."""
        return [item for item in self.nodes if node in self.nodes[item]]

    def topological_sort(self):
        """Topological sort."""
        in_degree = {node: 0 for node in self.nodes}
        out_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            out_degree[node] = len(node)
            for next_node in self.nodes[node]:
                in_degree[next_node] += 1
        ret = []
        stack = deque()
        for node in in_degree:
            if in_degree[node] == 0:
                stack.append(node)
        while len(stack) > 0:
            node = stack.pop()
            for item in self.nodes[node]:
                in_degree[item] -= 1
                if in_degree[item] == 0:
                    stack.append(item)
            ret.append(node)
        if len(ret) != len(self.nodes):
            raise ValueError("Not a directed acyclic graph")
        return ret

    def ind_nodes(self):
        """Independent nodes."""
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for next_node in self.nodes[node]:
                in_degree[next_node] += 1
        ret = set(node for node in self.nodes if in_degree[node] == 0)
        return ret

    def size(self):
        """Return the size of graph."""
        return len(self.nodes)
