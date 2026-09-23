
import random

from src.data import POPULATIONS
from src.tools import search, compare, calculator


def conversation(messages):
    for m in messages:
        m["content"] = str(m["content"])
    return {"messages": messages}


LOOKUP_TEMPLATES = [
    "What is the population of {country}?",
    "How many people live in {country}?",
    "Tell me the population of {country}.",
    "Can you find the population of {country}?",
]


def generate_lookup():

    country = random.choice(list(POPULATIONS.keys()))

    key = f"population_{country}"
    value = search(key)

    prompt = random.choice(LOOKUP_TEMPLATES).format(
        country=country.title()
    )

    return conversation([
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key}')</tool_call>"
        },
        {
            "role": "tool",
            "content": value
        },
        {
            "role": "assistant",
            "content": f"The population of {country.title()} is {value}."
        }
    ])


COMPARE_TEMPLATES = [
    "Which country has the larger population, {a} or {b}?",
    "Compare the populations of {a} and {b}.",
    "Between {a} and {b}, which is more populous?",
    "Determine which country has more people: {a} or {b}.",
]


COMPARE_ANSWERS = [
    "{winner} has the larger population.",
    "{winner} is more populous.",
    "The larger population is {winner}.",
    "{winner} has more people.",
]


def generate_compare():

    a, b = random.sample(list(POPULATIONS.keys()), 2)

    if random.random() < 0.5:
        first, second = a, b
    else:
        first, second = b, a

    key1 = f"population_{first}"
    key2 = f"population_{second}"

    value1 = search(key1)
    value2 = search(key2)

    larger = compare(value1, value2)

    winner = first if larger == value1 else second

    prompt = random.choice(COMPARE_TEMPLATES).format(
        a=first.title(),
        b=second.title()
    )

    answer = random.choice(COMPARE_ANSWERS).format(
        winner=winner.title()
    )

    return conversation([
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key1}')</tool_call>"
        },
        {
            "role": "tool",
            "content": value1
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key2}')</tool_call>"
        },
        {
            "role": "tool",
            "content": value2
        },
        {
            "role": "assistant",
            "content": f"<tool_call>compare({value1},{value2})</tool_call>"
        },
        {
            "role": "tool",
            "content": larger
        },
        {
            "role": "assistant",
            "content": answer
        }
    ])


COMPARE_MULTIPLY_TEMPLATES = [
    "Find the larger population of {a} and {b}, then multiply it by {m}.",
    "Which country has the larger population, {a} or {b}? Multiply that population by {m}.",
    "Compare the populations of {a} and {b}, then multiply the larger one by {m}.",
    "Determine which country has more people between {a} and {b}, then multiply that population by {m}.",
]


COMPARE_MULTIPLY_ANSWERS = [
    "The answer is {result}.",
    "The result is {result}.",
    "Multiplying the larger population by {m} gives {result}.",
    "The larger population multiplied by {m} is {result}.",
]


def generate_compare_multiply():

    a, b = random.sample(list(POPULATIONS.keys()), 2)

    multiplier = random.randint(2, 10)

    if random.random() < 0.5:
        first, second = a, b
    else:
        first, second = b, a

    key1 = f"population_{first}"
    key2 = f"population_{second}"

    value1 = search(key1)
    value2 = search(key2)

    larger = compare(value1, value2)

    result = calculator(f"{larger} * {multiplier}")

    prompt = random.choice(COMPARE_MULTIPLY_TEMPLATES).format(
        a=first.title(),
        b=second.title(),
        m=multiplier,
    )

    answer = random.choice(COMPARE_MULTIPLY_ANSWERS).format(
        result=result,
        m=multiplier,
    )

    return conversation([
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key1}')</tool_call>",
        },
        {
            "role": "tool",
            "content": value1,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key2}')</tool_call>",
        },
        {
            "role": "tool",
            "content": value2,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>compare({value1},{value2})</tool_call>",
        },
        {
            "role": "tool",
            "content": larger,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>calculator('{larger} * {multiplier}')</tool_call>",
        },
        {
            "role": "tool",
            "content": result,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ])


ADD_TEMPLATES = [
    "What is the combined population of {a} and {b}?",
    "Add the populations of {a} and {b}.",
    "Find the total population of {a} and {b}.",
    "What is the sum of the populations of {a} and {b}?",
]


ADD_ANSWERS = [
    "The combined population is {result}.",
    "The total population is {result}.",
    "The answer is {result}.",
    "{a} and {b} together have {result} people.",
]


def generate_add():

    a, b = random.sample(list(POPULATIONS.keys()), 2)

    if random.random() < 0.5:
        first, second = a, b
    else:
        first, second = b, a

    key1 = f"population_{first}"
    key2 = f"population_{second}"

    value1 = search(key1)
    value2 = search(key2)

    result = calculator(f"{value1} + {value2}")

    prompt = random.choice(ADD_TEMPLATES).format(
        a=first.title(),
        b=second.title(),
    )

    answer = random.choice(ADD_ANSWERS).format(
        a=first.title(),
        b=second.title(),
        result=result,
    )

    return conversation([
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key1}')</tool_call>",
        },
        {
            "role": "tool",
            "content": value1,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key2}')</tool_call>",
        },
        {
            "role": "tool",
            "content": value2,
        },
        {
            "role": "assistant",
            "content": f"<tool_call>calculator('{value1} + {value2}')</tool_call>",
        },
        {
            "role": "tool",
            "content": result,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ])


def generate_binding():

    a, b = random.sample(list(POPULATIONS.keys()), 2)

    if random.random() < 0.5:
        first, second = a, b
    else:
        first, second = b, a

    key1 = f"population_{first}"
    key2 = f"population_{second}"

    value1 = search(key1)
    value2 = search(key2)

    larger = compare(value1, value2)

    return conversation([
        {
            "role": "user",
            "content": f"Which country has a larger population, {first.title()} or {second.title()}?"
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key1}')</tool_call>"
        },
        {
            "role": "tool",
            "content": value1
        },
        {
            "role": "assistant",
            "content": f"<tool_call>search('{key2}')</tool_call>"
        },
        {
            "role": "tool",
            "content": value2
        },
        {
            "role": "assistant",
            "content": f"<tool_call>compare({value1},{value2})</tool_call>"
        },
        {
            "role": "tool",
            "content": larger
        },
        {
            "role": "assistant",
            "content": (
                f"{first.title()} and {second.title()} compared. "
                f"The larger population is "
                f"{first.title() if larger == value1 else second.title()}."
            )
        }
    ])


TASKS = [
    (generate_lookup, 0.15),
    (generate_compare, 0.25),
    (generate_compare_multiply, 0.25),
    (generate_add, 0.10),
    (generate_binding, 0.25),
]


def generate_task():
    funcs = [f for f, _ in TASKS]
    weights = [w for _, w in TASKS]

    return random.choices(funcs, weights=weights, k=1)[0]()


def trajectory_to_states(messages):
    examples = []

    for i, msg in enumerate(messages):

        if msg["role"] != "assistant":
            continue

        examples.append({
            "messages": messages[:i] + [
                {
                    "role": "assistant",
                    "content": msg["content"]
                }
            ]
        })

    return examples


def generate_dataset(n_trajectories=70_000):
    all_states = []

    for _ in range(n_trajectories):
        trajectory = generate_task()
        states = trajectory_to_states(trajectory["messages"])
        all_states.extend(states)

    return all_states
