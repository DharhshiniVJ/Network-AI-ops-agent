import re

path = "dashboard/app.py"
with open(path, "r") as f:
    text = f.read()

topology_section = """
st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE TOPOLOGIES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='the-topologies'></a>", unsafe_allow_html=True)
st.markdown('''
<div class='section-label'>03.5 — The Architectures</div>
<div class='section-title'>Enterprise WAN & Datacenter Spine-Leaf.</div>
<div class='section-body'>
We utilised two distinct network topologies: the <strong>Hierarchical Enterprise WAN</strong> 
(used for generating the massive simulated baseline dataset) and the 
<strong>Datacenter Spine-Leaf</strong> (used for live Mininet Linux-kernel emulation).
</div>
''', unsafe_allow_html=True)

import streamlit.components.v1 as components

mermaid_html = '''
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'dark', flowchart: { useMaxWidth: true } });
</script>
<style>
  body { background-color: #0f172a; margin: 0; padding: 20px; font-family: sans-serif; }
  .diagram-container { display: flex; flex-direction: column; align-items: center; gap: 60px; }
  .diagram-title { color: #f1f5f9; font-weight: 700; font-size: 1.1rem; text-align: center; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
<div class="diagram-container">
  <div style="width: 100%;">
    <div class="diagram-title">Datacenter Topology (Live Mininet Emulation)</div>
    <div class="mermaid" style="text-align: center;">
      flowchart TD
          classDef spine fill:#f97316,stroke:#c2410c,stroke-width:2px,color:white,font-weight:bold;
          classDef leaf fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:white,font-weight:bold;
          classDef host fill:#10b981,stroke:#047857,stroke-width:2px,color:white;
          
          S1[Spine 1]:::spine
          S2[Spine 2]:::spine
          L1[Leaf 1]:::leaf
          L2[Leaf 2]:::leaf
          L3[Leaf 3]:::leaf
          L4[Leaf 4]:::leaf
          H1((Host 1)):::host
          H2((Host 2)):::host
          H3((Host 3)):::host
          H4((Host 4)):::host
          
          S1 ===|40 Gbps| L1
          S1 ===|40 Gbps| L2
          S1 ===|40 Gbps| L3
          S1 ===|40 Gbps| L4
          S2 ===|40 Gbps| L1
          S2 ===|40 Gbps| L2
          S2 ===|40 Gbps| L3
          S2 ===|40 Gbps| L4
          
          L1 --- H1
          L2 --- H2
          L3 --- H3
          L4 --- H4
    </div>
  </div>

  <div style="width: 100%;">
    <div class="diagram-title">Enterprise WAN Topology (Simulated Training Data)</div>
    <div class="mermaid" style="text-align: center;">
      flowchart TD
          classDef core fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white,font-weight:bold;
          classDef dist fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:white,font-weight:bold;
          classDef access fill:#06b6d4,stroke:#0e7490,stroke-width:2px,color:white,font-weight:bold;
          
          C1{Core 1}:::core <==>|100 Gbps Link| C2{Core 2}:::core
          D1[Distribution 1]:::dist
          D2[Distribution 2]:::dist
          D3[Distribution 3]:::dist
          A1(Access 1):::access
          A2(Access 2):::access
          A3(Access 3):::access
          A4(Access 4):::access
          
          C1 ===|10 Gbps| D1
          C1 ===|10 Gbps| D2
          C1 ===|10 Gbps| D3
          C2 ===|10 Gbps| D1
          C2 ===|10 Gbps| D2
          C2 ===|10 Gbps| D3
          
          D1 ---|1 Gbps| A1
          D1 ---|1 Gbps| A2
          D2 ---|1 Gbps| A2
          D2 ---|1 Gbps| A3
          D3 ---|1 Gbps| A3
          D3 ---|1 Gbps| A4
    </div>
  </div>
</div>
'''
components.html(mermaid_html, height=1100, scrolling=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE ANOMALIES"""

# Replace `# THE ANOMALIES` with the new section
text = text.replace("# ══════════════════════════════════════════════════════════════════════════════\n# THE ANOMALIES", topology_section)

# Update sidebar link
sidebar_new = """st.markdown("<a href='#the-dataset' style='display:block; padding:10px 15px; color:#F1F5F9; text-decoration:none; font-weight:600; border-radius:8px; margin-bottom:5px; transition:0.2s;' onmouseover=\"this.style.background='#334155'\" onmouseout=\"this.style.background='transparent'\">03. The Dataset</a>", unsafe_allow_html=True)
    st.markdown("<a href='#the-topologies' style='display:block; padding:10px 15px; color:#F1F5F9; text-decoration:none; font-weight:600; border-radius:8px; margin-bottom:5px; transition:0.2s;' onmouseover=\"this.style.background='#334155'\" onmouseout=\"this.style.background='transparent'\">03.5. The Topologies</a>", unsafe_allow_html=True)"""
text = text.replace("st.markdown(\"<a href='#the-dataset' style='display:block; padding:10px 15px; color:#F1F5F9; text-decoration:none; font-weight:600; border-radius:8px; margin-bottom:5px; transition:0.2s;' onmouseover=\\\"this.style.background='#334155'\\\" onmouseout=\\\"this.style.background='transparent'\\\">03. The Dataset</a>\", unsafe_allow_html=True)", sidebar_new)


with open(path, "w") as f:
    f.write(text)
