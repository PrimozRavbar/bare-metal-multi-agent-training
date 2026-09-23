
import ast
import operator

from src.data import POPULATIONS


def search(key):
    return POPULATIONS[key.replace("population_", "")]


def compare(a, b):
    return max(a, b)


def calculator(expr):
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            op = allowed[type(node.op)]
            return op(_eval(node.left), _eval(node.right))

        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))
