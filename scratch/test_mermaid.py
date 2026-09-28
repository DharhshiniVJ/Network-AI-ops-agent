import streamlit as st
import streamlit.components.v1 as components

st.title("Mermaid Test")

mermaid_html = """
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'dark' });
</script>
<div class="mermaid" style="display:flex; justify-content:center;">
flowchart TD
    A[Hard] -->|Text| B(Round)
    B --> C{Decision}
    C -->|One| D[Result 1]
    C -->|Two| E[Result 2]
</div>
"""

components.html(mermaid_html, height=400)
