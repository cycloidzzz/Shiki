from typing import Dict, List, Tuple, Optional, Union
from abc import ABC

import numpy as np

from shiki.utils import is_scalar_type

ScalarType = Union[np.ndarray, float]

"""
Computation Node in the Computation Graph.
"""
class Node(object):
    def __init__(self,
                 name : Optional[str] = None):
        self.input_vals : List["Node"] = []
        self.op : Optional["Operation"] = None
        self.const_attr : Optional[ScalarType] = None

        # TODO(cycloidzzz) : a reserved field `name` for debugging use?
        self.name = "Node "
        if name is not None:
            self.name = name

    def __add__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return add_const_op(self, rhs)
        else:
            return add_op(self, rhs)

    def __sub__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return add_const_op(self, -1 * rhs)
        else:
            return add_op(self, mul_const_op(rhs, -1))

    def __rsub__(self, lhs) -> "Node":
        if is_scalar_type(lhs):
            return add_const_op(mul_const_op(self, -1), lhs)
        else:
            return add_op(lhs, mul_const_op(self, -1))

    def __mul__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return mul_const_op(self, rhs)
        else:
            return mul_op(self, rhs)

    def __div__(self, rhs) -> "Node":
        if is_scalar_type(rhs):
            return mul_const_op(self, 1/rhs)
        else:
            return divide_op(self, rhs)

    def __rdiv__(self, lhs) -> "Node":
        if is_scalar_type(lhs):
            return mul_const_op(divide_op(ones_like_op(self), sefl), lhs)
        else:
            return divide_op(lhs, self)

    __radd__ = __add__
    __rmul__ = __mul__


"""
Base class for Tensor Operation of Nodes, which does not hold any data.
    """
class Operation(ABC):
    def __call__(self, name: Optional[str] = None):
        new_node : Node = Node(name=name)
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
        new_node : Node = Operation.__call__(self, name = name)
        
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


class OnesLikeOperation(Operation):
    def __call__(self,
                 node_a : Node) -> Node:
        new_name : str = f"OnesLike({node_a.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals : List[Node] = [node_a]
        
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return np.ones_like(input_vals[0])

    def gradient(self,
                ctx : Node,
                out_grad : Node) -> List[Node]:
        node_a, = ctx.input_vals
        return [zeros_like_op(node_a)]
        

class ZerosLikeOperation(Operation):
    def __call__(self,
                node_a : Node) -> Node:
        new_name : str = f"ZerosLike({node_a.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_ndoe.input_vals : List[Node] = [node_a]
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return np.zeros_like(input_vals[0])

    def gradient(self,
                ctx : Node,
                out_grad : Node) -> List[Node]:
        node_a, = ctx.input_vals
        return [zeros_like_op(node_a)]


class AddOperation(Operation):
    def __call__(self, 
                 node_a : Node,
                 node_b : Node) -> Node:
        new_name : str = f"({node_a.name} + {node_b.name})"
        new_node : Node = Operation.__call__(self, name = new_name)
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
        name : str = f"({node_a.name} + {const_attr})"
        new_node : Node = Operation.__call__(self, name=name)
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
        new_name : str = f"({node_a.name} * {node_b.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
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
        new_name : str = f"({node_a.name} * {const_attr})"
        new_node : Node = Operation.__call__(self, name=new_name)
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
        new_name : str = f"{node_a.name}.transpose()"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a]

        return new_node
    
    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        return input_vals[0].transpose()

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        return [transpose_op(out_grad)]


class MatMulOperation(Operation):
    def __call__(self, node_a : Node, node_b : Node) -> Node:
        new_name : str = f"matmul({node_a.name}, {node_b.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
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


# TODO (cycloidz) : Softmax along arbitrary dimension.
class SoftmaxOperation(Operation):
    def __call__(self, 
                 node_a : Node) -> Node:
        new_name : str = f"Softmax({node_a.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a]
        return new_node
    
    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        assert len(input_vals) == 1, "SoftmaxOperation : len of input list is not 1."
        a_val : ScalarType = input_vals[0]
        # TODO (cycloidz) : Substract min(or max?) along the dimension
        # in case of underflow.
        a_exp : ScalarType = np.exp(a_val)
        a_dominator : ScalarType = np.sum(a_exp, axis=-1, keepdims=True)
        a_softmax : ScalarType = a_exp / a_dominator
        return  a_softmax

    def gradient(self,
                ctx : Node,
                out_grad : Node) -> List[Node]:
        node_a : Node = ctx.input_vals[0]
        node_softmax : Node = softmax_op(node_a)
        node_sum : Node = reduce_sum_op(out_grad * node_softmax, keepdims=True)
        node_grad : Node = (out_grad - node_sum) * node_softmax
        return [node_grad]

class SoftmaxCrossEntropyOperation(Operation):
    def __call__(self, 
                logits : Node, 
                labels : Node) -> Node:
        new_name : str = f"SoftmaxCrossEntropy({logits.name}, {labels.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [logits, labels]
        
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        val_logits : ScalarType = input_vals[0]
        val_labels : ScalarType = input_vals[1]
        
        val_log_softmax : ScalarType = val_logits - \
            np.log(np.sum(np.exp(val_logits), axis=-1, keepdims=True))
        return - val_log_softmax * val_labels

    def gradient(self, 
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        logits : Node = ctx.input_vals[0]
        labels : Node = ctx.input_vals[1]

        grad_logits : Node = labels * -1  +  reduce_sum_op(labels, keepdims=True) * softmax_op(logits)
        grad_labels : Node = log_op(softmax_op(logits)) * -1
        return [grad_logits, grad_labels]


class LogOperation(Operation):
    def __call__(self, node_a : Node) -> Node:
        new_name : str = f"Log({node_a.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a]
        
        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        val_a : ScalarType = input_vals[0]
        return np.log(val_a)

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        node_a : Node = input_vals[0]
        return [divide_op(ones_line_op(node_a), node_a)]


class DivideOperation(Operation):
    def __call__(self, node_a : Node, node_b : Node) -> Node:
        new_name : str = f"({node_a.name}/{node_b.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a, node_b]

        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        val_a : ScalarType = input_vals[0]
        val_b : ScalarType = input_vals[1]
        return val_a / val_b

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        node_a, node_b = ctx.input_vals
        grad_a : Node = out_grad / node_b
        grad_b : Node = out_grad * (node_a / (node_b * node_b)) * -1
        return [grad_a, grad_b]


class ReduceSumOperation(Operation):
    def __call__(self, node_a : Node, keepdims : bool = True) -> Node:
        # TODO(cycloidz) : the case keepdims = False.
        assert keepdims == True, "ReduceSumOperation : keepdims == False is not supported so far."
        new_name : str = f"ReduceSum({node_a.name})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a]
        new_node.keepdims = True

        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        val_a : ScalarType = input_vals[0]
        return np.sum(val_a, axis=-1, keepdims=ctx.keepdims)

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        node_a : Node = ctx.input_vals[0]
        node_sum : Node = reduce_sum_op(out_grad, keepdims=True)
        return [node_sum]


class ExpandDimsOperation(Operation):
    def __call__(self, 
                 node_a : Node,
                 axis : int = -1) -> Node:
        new_name : str = f"ExpandDims({node_a.name})"
        new_node : Node = Operation.__call__(self, name = new_name)
        new_node.input_vals = [node_a]
        new_node.axis = axis

        return new_node

    def compute(self, 
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        assert len(input_vals) == 1, "ExpandDims : input list should contain exactly one value."
        val_a : ScalarType = input_vals[0]
        return np.expand_dims(val_a, axis=ctx.axis)

    def gradient(self,
                 ctx : Node,
                 out_grad : Node) -> List[Node]:
        grad_a : Node = squeeze_op(out_grad, axis=ctx.axis)
        return [grad_a]


class SqueezeOperation(Operation):
    def __call__(self, 
                 node_a : Node,
                 axis : int = -1) -> Node:
        new_name : str = f"Squeeze({node_a.name}, axis={axis})"
        new_node : Node = Operation.__call__(self, name=new_name)
        new_node.input_vals = [node_a]
        new_node.axis = axis

        return new_node

    def compute(self,
                ctx : Node,
                input_vals : List[ScalarType]) -> ScalarType:
        assert len(input_vals) == 1, "Squeeze : len(input_vals) should be 1."
        val_a : ScalarType = input_vals[0]
        return np.squeeze(val_a, axis=ctx.axis)

    def gradient(self,
                ctx : Node,
                out_grad : Node) -> List[Node]:
        grad_a : Node = expand_dims_op(out_grad, axis=ctx.axis)
        return [grad_a]


def Variable(name : str) -> Node:
    new_node = placeholder_op(name)
    return new_node

placeholder_op = PlaceholderOperation()
zeros_like_op = ZerosLikeOperation()
ones_like_op = OnesLikeOperation()
add_op = AddOperation()
add_const_op = AddByConstOperation()
mul_op = MulOperation()
mul_const_op = MulByConstOperation()
divide_op = DivideOperation()
matmul_op = MatMulOperation()
transpose_op = TransposeOperation()
softmax_op = SoftmaxOperation()
softmax_cross_entropy_with_logits_op = SoftmaxCrossEntropyOperation()
log_op = LogOperation()
reduce_sum_op = ReduceSumOperation()
expand_dims_op = ExpandDimsOperation()
squeeze_op = SqueezeOperation()

class Executor(object):
    def __init__(self, 
                 eval_node_list : List[Node]):
        self.eval_node_list : List[Node] = eval_node_list
        self.topo_sort_list : List[Node] = topology_sort(eval_node_list)
    
    def run(self, feed_dict : Dict[Node, ScalarType]) -> List[ScalarType]:
        node_result_map : Dict[Node, ScalarType] = {}
        for (node, node_val) in feed_dict.items():
            node_result_map[node] = node_val
        
        for node in self.topo_sort_list:
            if node in node_result_map:
                continue
            input_vals : List[ScalarType] = [
                node_result_map[pred] for pred in node.input_vals
            ]
            #print(f"forwarding : {node.name}")
            #print(input_vals)
            node_val : ScalarType = node.op.compute(node, input_vals)
            node_result_map[node] = node_val
        
        eval_result_list : List[ScalarType] = [
            node_result_map[node]
            for node in self.eval_node_list
        ]
        
        return eval_result_list
            

def topology_sort(node_list : List[Node]) -> List[Node]:
    visited : Dict[Node, bool] = {}
    toposort_res : List[Node] = []

    def dfs_topology_sort(node : Node,
                        visited_map : Dict[Node, bool],
                        topo_list : List[Node]):
        if node in visited_map:
            return
        visited_map[node] = True
        for pred in node.input_vals:
            dfs_topology_sort(pred, visited_map, topo_list)
        topo_list.append(node)

    for node in node_list:
        dfs_topology_sort(node, visited, toposort_res)
    
    return toposort_res


def gradient(out_node : Node,
             output_list : List[Node]) -> List[Node] :

    node_grad_list_map : Dict[Node, List[Node]] = {}
    node_grad_map : Dict[Node, Node] = {}

    node_grad_list_map[out_node] = [ones_like_op(out_node)]

    vertices : List[Node] = list(reversed(topology_sort([out_node])))
    
    for node in vertices:
        out_grad : Node = grad_list_reduce(node_grad_list_map[node])
        node_grad_map[node] = out_grad

        if isinstance(node.op, PlaceholderOperation):
            continue

        input_grads : List[Node] = node.op.gradient(node, out_grad)

        for (pred, input_grad) in zip(node.input_vals, input_grads):
            if pred not in node_grad_list_map:
                node_grad_list_map[pred] = [input_grad]
            else:
                node_grad_list_map[pred].append(input_grad)

    grad_output_list : List[Node] = [
        node_grad_map[node] for node in output_list
    ]
    return grad_output_list


def grad_list_reduce(grad_list : List[Node]) -> Node:
    from operator import add
    from functools import reduce
    return reduce(add, grad_list)
