
import re
import ast
import operator

from src.tools import search, compare
from src.model import generate


def calculate(expr):
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
            return allowed[type(node.op)](
                _eval(node.left),
                _eval(node.right)
            )

        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))


def run_agent(user_prompt, model, tokenizer, max_steps=10):

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    last_tool = None
    last_result = None

    for step in range(max_steps):

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output = generate(
            model,
            tokenizer,
            prompt
        )

        if output.startswith(prompt):
            output = output[len(prompt):]

        if "<|im_end|>" in output:
            output = output.split("<|im_end|>")[0]

        output = output.strip()

        print("RAW:", output)

        if "<tool_call>" not in output:
            print("FINAL:", output)
            return output

        tool_call = (
            output
            .split("<tool_call>")[1]
            .split("</tool_call>")[0]
            .strip()
        )

        print("MODEL:", tool_call)

        # prevent infinite identical tool loops
        if tool_call == last_tool and str(last_result) in tool_call:
            print("Repeated tool call. Stopping.")
            break

        # SEARCH
        if tool_call.startswith("search"):

            match = re.search(
                r"'([^']+)'",
                tool_call
            )

            if not match:
                break

            result = search(
                match.group(1)
            )

        # COMPARE
        elif tool_call.startswith("compare"):

            nums = re.findall(
                r"\d+",
                tool_call
            )

            if len(nums) != 2:
                break

            result = compare(
                int(nums[0]),
                int(nums[1])
            )

        # CALCULATOR
        elif tool_call.startswith("calculator"):

            match = re.search(
                r"'([^']+)'",
                tool_call
            )

            if not match:
                break

            result = calculate(
                match.group(1)
            )

        else:
            print("Unknown tool:", tool_call)
            break

        print("TOOL:", result)

        last_tool = tool_call
        last_result = result

        messages.append(
            {
                "role": "assistant",
                "content":
                    f"<tool_call>{tool_call}</tool_call>"
            }
        )

        messages.append(
            {
                "role": "tool",
                "content": str(result)
            }
        )

    return output
