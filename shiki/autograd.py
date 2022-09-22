from typing import Dict, List, Tuple, Optional, Union
from abc import ABC

import numpy as np

from shiki.utils import is_scalar_type

ScalarType = Union[np.ndarray, float]

"""
Computation Node in the Computation Graph.
"""
class Node(object):
    def __init__(self):
        self.input_vals : List["Node"] = []
        self.op : Optional["Operation"] = None
        self.const_attr : Optional[ScalarType] = None

        # TODO(cycloidzzz) : a reserved field `name` for debugging use?

    def __add__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return add_const_op(self, rhs)
        else:
            return add_op(self, rhs)

    def __mul__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return add_const_op(self, rhs)
        else:
            return add_op(self, rhs)

    __radd__ = __add__
    __rmul__ = __mul__


"""
Base class for Tensor Operation of Nodes, which does not hold any data.
"""
class Operation(ABC):
    def __call__(self):
        new_node : Node = Node()
        new_node.op = self
        return new_node
    
    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        raise NotImplementedError("Compute Method is not implemented.")

    def gradient(self,
                 ctx : Node,
                 out_grads : Node) -> List[Node]:
        raise NotImplementedError("Gradient Method is not implemented.")


class PlaceholderOperation(Operation):
    def __call__(self, 
                 name : str) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.name = name
        
        return new_node
    
    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        assert len(input_vals) == 1, "Placeholder Op should contain exactly one input value."
        return input_vals[0]

    def gradient(self, 
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        assert false, "Placeholder Op : Cannot calculate the gradient value for Placeholder Operation."


class AddOperation(Operation):
    def __call__(self, 
                 node_a : Node,
                 node_b : Node) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a, node_b]
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0] + input_vals[1]

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        return [out_grad, out_grad]


class AddByConstOperation(Operation):
    def __call__(self, node_a : Node, const_attr : ScalarType) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a]
        new_node.const_attr = const_attr
        return new_node

    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0] + ctx.const_attr

    def gradient(self, 
                 ctx: Node,
                 out_grad : Node) -> ScalarType:
        return [out_grad]


class MulOperation(Operation):
    def __call__(self, node_a : Node, node_b : Node) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a, node_b]

        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0] * input_vals[1]

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        node_a, node_b = ctx.input_vals
        out_grad_a : Node = node_b * out_grad
        out_grad_b : Node = node_a * out_grad
        return [out_grad_a, out_grad_b]


class MulByConstOperation(Operation):
    def __call__(self, node_a : Node, const_attr : ScalarType) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a]
        new_node.const_attr = const_attr

        return new_node

    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0] * ctx.const_attr

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        return [out_grad * ctx.const_attr]


class TransposeOperation(Operation):
    def __call__(self, node_a : Node) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a]

        return new_node
    
    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0].transpose()

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        return [out_grad.transpose()]


class MatMulOperation(Operation):
    def __call__(self, node_a : Node, node_b : Node) -> Node:
        new_node : Node = Operation.__call__(self)
        new_node.input_vals = [node_a, node_b]
        
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return np.matmul(input_vals[0], input_vals[1])

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        node_a, node_b = ctx.input_vals[0], ctx.input_vals[1]
        # (M, K), (K, N)
        grad_a : Node = matmul_op(out_grad, transpose_op(node_b))
        grad_b : Node = matmul_op(transpose_op(node_a), out_grad)
        return [grad_a, grad_b]

def Variable(name : str) -> Node:
    new_node = placeholder_op(name)
    return new_node

placeholder_op = PlaceholderOperation()
add_op = AddOperation()
add_const_op = AddByConstOperation()
mul_op = MulOperation()
mul_const_op = MulByConstOperation()
matmul_op = MatMulOperation()
transpose_op = TransposeOperation()