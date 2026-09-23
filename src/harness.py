import re

from src.tools import search, compare, calculator
from src.model import generate


def run_agent(user_prompt, model, tokenizer, max_steps=10):
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    last_tool = None
    last_result = None

    for step in range(max_steps):

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = generate(
            model,
            tokenizer,
            prompt,
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
            output.split("<tool_call>")[1]
            .split("</tool_call>")[0]
            .strip()
        )

        print("MODEL:", tool_call)

        if tool_call == last_tool and str(last_result) in tool_call:
            print("Repeated tool call. Stopping.")
            break

        if tool_call.startswith("search"):
            match = re.search(r"'([^']+)'", tool_call)

            if not match:
                break

            result = search(match.group(1))

        elif tool_call.startswith("compare"):
            nums = re.findall(r"\d+", tool_call)

            if len(nums) != 2:
                break

            result = compare(
                int(nums[0]),
                int(nums[1]),
            )

        elif tool_call.startswith("calculator"):
            match = re.search(r"'([^']+)'", tool_call)

            if not match:
                break

            result = calculator(match.group(1))

        else:
            print("Unknown tool:", tool_call)
            break

        print("TOOL:", result)

        last_tool = tool_call
        last_result = result

        messages.append({
            "role": "assistant",
            "content": f"<tool_call>{tool_call}</tool_call>",
        })

        messages.append({
            "role": "tool",
            "content": str(result),
        })

    return output
