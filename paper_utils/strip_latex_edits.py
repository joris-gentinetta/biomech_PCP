import re
import sys
from pathlib import Path


# Remove all \del{...} blocks, including nested braces
def remove_del_blocks(s):
    pattern = re.compile(r"\\del\{")
    out = []
    i = 0
    while i < len(s):
        m = pattern.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i : m.start()])
        # Find matching closing brace
        depth = 1
        j = m.end()
        while j < len(s) and depth > 0:
            if s[j : j + 5] == "\\del{":
                depth += 1
                j += 5
            elif s[j] == "{":
                depth += 1
                j += 1
            elif s[j] == "}":
                depth -= 1
                j += 1
            else:
                j += 1
        i = j
    return "".join(out)


# Replace all \add{...} with just the content inside, including nested braces
def unwrap_add_blocks(s):
    pattern = re.compile(r"\\add\{")
    out = []
    i = 0
    while i < len(s):
        m = pattern.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i : m.start()])
        # Find matching closing brace
        depth = 1
        j = m.end()
        start_content = j
        while j < len(s) and depth > 0:
            if s[j : j + 5] == "\\add{":
                depth += 1
                j += 5
            elif s[j] == "{":
                depth += 1
                j += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
                j += 1
            else:
                j += 1
        out.append(s[start_content:j])
        j += 1  # skip closing '}'
        i = j
    return "".join(out)


def process_latex_edits(text):
    # Remove all \del{...}
    text = remove_del_blocks(text)
    # Replace all \add{...} with their content
    text = unwrap_add_blocks(text)
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strip_latex_edits.py <input_file> [<output_file>]")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    processed = process_latex_edits(text)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed)
    else:
        print(processed)
