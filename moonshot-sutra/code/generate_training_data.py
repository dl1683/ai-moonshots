"""Generate rich synthetic training data for Sutra probes.

Creates structured text with reasoning chains, hierarchical concepts,
compositional patterns, and varied complexity. CPU-bound.

This data will be used across all probes for consistency.
"""

import json
import os
import random
from pathlib import Path

SEED = 42
random.seed(SEED)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


def generate_arithmetic_chains(n=5000):
    """Multi-step arithmetic reasoning chains."""
    examples = []
    for _ in range(n):
        n_steps = random.randint(1, 6)
        val = random.randint(1, 50)
        chain = [f"Start with {val}."]
        for _ in range(n_steps):
            op = random.choice(["add", "subtract", "multiply by", "divide by"])
            operand = random.randint(1, 10)
            if op == "add":
                val += operand
            elif op == "subtract":
                val -= operand
            elif op == "multiply by":
                val *= operand
            else:
                if operand != 0:
                    val = val // operand
            chain.append(f"{op.capitalize()} {operand}.")
        chain.append(f"Result: {val}.")
        examples.append(" ".join(chain))
    return examples


def generate_logic_chains(n=5000):
    """If-then logical reasoning chains."""
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
             "Iris", "Jack", "Kate", "Leo", "Mia", "Nate", "Olive", "Pete"]
    properties = ["tall", "smart", "fast", "kind", "brave", "wise", "calm", "bold",
                  "fair", "warm", "keen", "mild", "firm", "glad", "neat", "rich"]

    examples = []
    for _ in range(n):
        n_steps = random.randint(2, 5)
        selected_names = random.sample(names, min(n_steps + 1, len(names)))
        selected_props = random.sample(properties, min(n_steps + 1, len(properties)))

        chain = []
        chain.append(f"{selected_names[0]} is {selected_props[0]}.")
        for i in range(n_steps):
            chain.append(f"If someone is {selected_props[i]}, they are {selected_props[i+1]}.")
        chain.append(f"Therefore, {selected_names[0]} is {selected_props[n_steps]}.")
        examples.append(" ".join(chain))
    return examples


def generate_hierarchical_descriptions(n=5000):
    """Hierarchical category descriptions (tests abstraction)."""
    hierarchy = {
        "animal": {
            "mammal": {"dog": ["barks", "loyal"], "cat": ["purrs", "independent"], "whale": ["large", "ocean"]},
            "bird": {"eagle": ["soars", "predator"], "sparrow": ["small", "common"], "penguin": ["flightless", "cold"]},
            "fish": {"salmon": ["swims upstream", "pink"], "shark": ["predator", "ancient"], "clown": ["colorful", "reef"]},
        },
        "vehicle": {
            "land": {"car": ["four wheels", "road"], "truck": ["large", "cargo"], "bicycle": ["pedals", "two wheels"]},
            "water": {"boat": ["floats", "water"], "submarine": ["underwater", "deep"], "canoe": ["paddles", "river"]},
            "air": {"airplane": ["flies", "wings"], "helicopter": ["rotors", "vertical"], "balloon": ["floats", "hot air"]},
        },
        "food": {
            "fruit": {"apple": ["red", "crunchy"], "banana": ["yellow", "soft"], "grape": ["small", "cluster"]},
            "vegetable": {"carrot": ["orange", "root"], "broccoli": ["green", "tree-like"], "potato": ["starchy", "underground"]},
            "grain": {"rice": ["white", "staple"], "wheat": ["golden", "bread"], "corn": ["yellow", "cob"]},
        },
    }

    examples = []
    for _ in range(n):
        # Pick a random item and describe it at multiple levels
        top = random.choice(list(hierarchy.keys()))
        mid = random.choice(list(hierarchy[top].keys()))
        item = random.choice(list(hierarchy[top][mid].keys()))
        props = hierarchy[top][mid][item]

        # Generate descriptions at different hierarchy levels
        style = random.choice(["top_down", "bottom_up", "comparative"])
        if style == "top_down":
            text = f"A {item} is a type of {mid}, which is a type of {top}. "
            text += f"It is known for being {props[0]} and {props[1]}."
        elif style == "bottom_up":
            text = f"Something that is {props[0]} and {props[1]} is a {item}. "
            text += f"A {item} belongs to the {mid} category of {top}s."
        else:
            # Compare two items in the same category
            other_item = random.choice([k for k in hierarchy[top][mid].keys() if k != item])
            other_props = hierarchy[top][mid][other_item]
            text = f"Both {item} and {other_item} are {mid}s. "
            text += f"The {item} is {props[0]} while the {other_item} is {other_props[0]}."

        examples.append(text)
    return examples


def generate_compositional_sentences(n=5000):
    """Sentences with compositional structure (tests composition)."""
    subjects = ["the tall man", "a small child", "the old woman", "a young dog",
                "the red car", "a blue bird", "the fast runner", "a slow turtle"]
    verbs = ["sees", "chases", "helps", "follows", "watches", "finds", "greets", "avoids"]
    objects = ["the green tree", "a big house", "the dark cave", "a bright light",
               "the cold river", "a warm fire", "the quiet garden", "a loud crowd"]
    modifiers = ["quickly", "slowly", "carefully", "happily", "sadly", "eagerly", "gently", "boldly"]
    connectors = ["and then", "because", "although", "while", "before", "after", "whenever", "unless"]

    examples = []
    for _ in range(n):
        n_clauses = random.randint(1, 4)
        clauses = []
        for _ in range(n_clauses):
            s = random.choice(subjects)
            v = random.choice(verbs)
            o = random.choice(objects)
            m = random.choice(modifiers) if random.random() > 0.5 else ""
            clause = f"{s} {m + ' ' if m else ''}{v} {o}"
            clauses.append(clause)

        if len(clauses) == 1:
            text = clauses[0] + "."
        else:
            parts = [clauses[0]]
            for c in clauses[1:]:
                conn = random.choice(connectors)
                parts.append(f"{conn} {c}")
            text = " ".join(parts) + "."
        examples.append(text.capitalize())
    return examples


def generate_pattern_sequences(n=5000):
    """Sequences with discoverable patterns (tests compression/prediction)."""
    examples = []
    for _ in range(n):
        pattern_type = random.choice(["repeat", "arithmetic", "fibonacci_like", "mirror", "alternating"])

        if pattern_type == "repeat":
            unit = [random.randint(1, 9) for _ in range(random.randint(2, 5))]
            repeats = random.randint(3, 6)
            seq = (unit * repeats)[:20]
            text = f"Pattern: {' '.join(map(str, seq))}. Next: {' '.join(map(str, unit[:3]))}."

        elif pattern_type == "arithmetic":
            start = random.randint(1, 20)
            step = random.randint(1, 5)
            length = random.randint(5, 10)
            seq = [start + i * step for i in range(length)]
            text = f"Sequence: {' '.join(map(str, seq))}. Next: {seq[-1] + step}."

        elif pattern_type == "fibonacci_like":
            a, b = random.randint(1, 5), random.randint(1, 5)
            seq = [a, b]
            for _ in range(6):
                seq.append(seq[-1] + seq[-2])
            text = f"Sequence: {' '.join(map(str, seq))}. Next: {seq[-1] + seq[-2]}."

        elif pattern_type == "mirror":
            half = [random.randint(1, 9) for _ in range(random.randint(3, 6))]
            seq = half + half[::-1]
            text = f"Palindrome: {' '.join(map(str, seq))}."

        else:  # alternating
            a, b = random.randint(1, 9), random.randint(1, 9)
            length = random.randint(6, 12)
            seq = [a if i % 2 == 0 else b for i in range(length)]
            text = f"Alternating: {' '.join(map(str, seq))}. Next: {a if length % 2 == 0 else b}."

        examples.append(text)
    return examples


def main():
    print("Generating rich synthetic training data for Sutra probes...")
    print(f"Using {os.cpu_count()} CPU cores")
    print("=" * 60)

    generators = [
        ("arithmetic_chains", generate_arithmetic_chains),
        ("logic_chains", generate_logic_chains),
        ("hierarchical_descriptions", generate_hierarchical_descriptions),
        ("compositional_sentences", generate_compositional_sentences),
        ("pattern_sequences", generate_pattern_sequences),
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_text = []

    for name, gen_fn in generators:
        print(f"  Generating {name}...", flush=True)
        examples = gen_fn(5000)
        all_text.extend(examples)
        print(f"    {len(examples)} examples, ~{sum(len(e) for e in examples):,} chars")

    # Shuffle and combine
    random.shuffle(all_text)
    full_text = "\n".join(all_text)

    # Split train/test
    split = int(len(all_text) * 0.9)
    train_text = "\n".join(all_text[:split])
    test_text = "\n".join(all_text[split:])

    # Save
    train_path = DATA_DIR / "train.txt"
    test_path = DATA_DIR / "test.txt"
    meta_path = DATA_DIR / "meta.json"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)

    meta = {
        "total_examples": len(all_text),
        "train_examples": split,
        "test_examples": len(all_text) - split,
        "train_chars": len(train_text),
        "test_chars": len(test_text),
        "categories": {name: 5000 for name, _ in generators},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATASET GENERATED")
    print(f"  Total: {len(all_text):,} examples")
    print(f"  Train: {split:,} examples, {len(train_text):,} chars")
    print(f"  Test:  {len(all_text)-split:,} examples, {len(test_text):,} chars")
    print(f"  Categories: {', '.join(name for name, _ in generators)}")
    print(f"  Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
