import re

path = "dashboard/pages/2_Results.py"
with open(path, "r") as f:
    text = f.read()

# Pattern: fig_something.update_layout(**PLOTLY_TEMPLATE['layout'], title='Something', ...)
# We want to split it:
# fig_something.update_layout(PLOTLY_TEMPLATE['layout'])
# fig_something.update_layout(title='Something', ...)

def replacer(match):
    fig_name = match.group(1)
    rest = match.group(2)
    return f"{fig_name}.update_layout(PLOTLY_TEMPLATE['layout'])\n        {fig_name}.update_layout({rest}"

text = re.sub(r'([a-zA-Z0-9_]+)\.update_layout\(\*\*PLOTLY_TEMPLATE\[\'layout\'\],\s*(.*?\))', replacer, text)

with open(path, "w") as f:
    f.write(text)

