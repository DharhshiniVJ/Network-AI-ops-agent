import re

path = "dashboard/pages/2_Results.py"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "update_layout(PLOTLY_TEMPLATE['layout'])" in line:
        new_lines.append(line)
        # the next line has incorrect indentation. The current line has the correct indentation.
        # we can't just fix it here blindly because the next line is processed next, 
        # wait, let me just fix the indentation of the entire file.
