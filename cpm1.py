import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# Fungsi untuk membaca file CSV
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="CPM (Critical Path Method)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fungsi utama untuk menghitung dan menampilkan CPM
def calculate_cpm(data):
    G = nx.DiGraph()
    all_nodes = set(data['Notasi'].tolist())

    # Tambahkan node ke grafik
    for _, row in data.iterrows():
        G.add_node(row['Notasi'],
                   duration=row['Durasi (Hari)'],
                   early_start=0,
                   early_finish=0,
                   late_start=float('inf'),
                   late_finish=float('inf'))

    # Tambahkan edge berdasarkan dependensi
    for _, row in data.iterrows():
        predecessors = str(row['Kegiatan Yang Mendahului']).split(',')
        for predecessor in predecessors:
            predecessor = predecessor.strip()
            if predecessor == '-' or predecessor == '':
                continue
            if predecessor in all_nodes:
                G.add_edge(predecessor, row['Notasi'])
            else:
                st.warning(f"Notasi '{predecessor}' tidak ditemukan dalam data. Dilewati. (Baris notasi: '{row['Notasi']}')")

    try:
        # FORWARD PASS
        for node in nx.topological_sort(G):
            early_start = max([G.nodes[pred]['early_finish'] for pred in G.predecessors(node)], default=0)
            G.nodes[node]['early_start'] = early_start
            G.nodes[node]['early_finish'] = early_start + G.nodes[node]['duration']

        # Durasi total proyek
        project_duration = max(G.nodes[n]['early_finish'] for n in G.nodes)

        # BACKWARD PASS
        for node in reversed(list(nx.topological_sort(G))):
            successors = list(G.successors(node))
            if not successors:
                G.nodes[node]['late_finish'] = project_duration
                G.nodes[node]['late_start'] = project_duration - G.nodes[node]['duration']
            else:
                min_ls = min([G.nodes[succ]['late_start'] for succ in successors])
                G.nodes[node]['late_finish'] = min_ls
                G.nodes[node]['late_start'] = min_ls - G.nodes[node]['duration']

        # Hitung slack
        for node in G.nodes:
            G.nodes[node]['Slack'] = G.nodes[node]['late_start'] - G.nodes[node]['early_start']

        # Jalur kritis
        critical_path = [n for n in nx.topological_sort(G) if G.nodes[n]['Slack'] == 0]
        critical_path_edges = [(critical_path[i], critical_path[i + 1]) for i in range(len(critical_path) - 1)]

        # Visualisasi
        for node in G.nodes:
            G.nodes[node]['level'] = G.nodes[node]['early_start']
        pos = nx.multipartite_layout(G, subset_key="level")

        label_full = {}
        for _, row in data.iterrows():
            node = row['Notasi']
            es = G.nodes[node]['early_start']
            ls = G.nodes[node]['late_start']
            label_full[node] = f"{node}\nES: {es}\nLS: {ls}"

        plt.figure(figsize=(20, 7))
        nx.draw_networkx_edges(G, pos, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=900, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, labels=label_full, font_size=6, font_weight='bold')

        for u, v in G.edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2 + 0.01
            durasi = G.nodes[u]['duration']
            plt.text(xm, ym, f"{durasi} hari", fontsize=6, fontweight='bold', color='blue', ha='center', va='bottom')

        nx.draw_networkx_edges(G, pos, edgelist=critical_path_edges, edge_color='red', width=2)

        plt.title(f'Critical Path: {" → ".join(critical_path)} | Durasi Total: {project_duration} hari', fontsize=10)
        plt.axis('off')
        st.pyplot(plt)

        # Tampilkan tabel hasil
        st.subheader("Tabel Hasil CPM")
        hasil = []
        for node in G.nodes:
            n = node
            es = G.nodes[node]['early_start']
            ef = G.nodes[node]['early_finish']
            ls = G.nodes[node]['late_start']
            lf = G.nodes[node]['late_finish']
            slack = G.nodes[node]['Slack']
            hasil.append([n, es, ef, ls, lf, slack])
        df_result = pd.DataFrame(hasil, columns=['Node', 'ES', 'EF', 'LS', 'LF', 'Slack'])
        st.dataframe(df_result)

    except nx.NetworkXUnfeasible:
        st.error("Struktur grafik tidak valid. Mungkin ada siklus atau kesalahan notasi.")

# Sidebar
st.sidebar.header('Upload File CSV')
uploaded_file = st.sidebar.file_uploader("Upload file CPM (format CSV)", type=["csv"])

st.title("📊 Aplikasi CPM (Critical Path Method)")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("Data Aktivitas Proyek")
    st.dataframe(df)
    calculate_cpm(df)
else:
    st.info("Silakan upload file CSV untuk memulai.")
