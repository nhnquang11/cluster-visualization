import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utility import *


@st.cache_data
def load_data():
    wp2 = pd.read_csv('data/wallet_profile_2.csv')
    w2w = pd.read_csv('data/wallet_to_wallet.csv')
    ethw = pd.read_csv('data/eth_wallets.csv')
    ethsc = pd.read_csv('data/ETH_smart_contract.csv')
    w2sc = pd.read_csv('data/w2sc.csv')
    return wp2, w2w, ethw, ethsc, w2sc


if 'address_input' not in st.session_state:
    st.session_state['address_input'] = '0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43'

wp2, w2w, ethw, ethsc, w2sc = load_data()

st.title("Visualization")

random_butt = st.button("Random Address", type="secondary")
if random_butt:
    st.session_state['address_input'] = get_random_address(wp2)
    st.text_input("Address input", value=st.session_state['address_input'])
else:
    st.session_state['address_input'] = st.text_input(
        "Address input", st.session_state['address_input'])

periods = st.number_input(
    "Period (days): ", min_value=0, step=1, value=10, format="%d")
k_top_wallets = st.number_input(
    "Number of top wallets to show: ", min_value=0, step=1, value=5, format="%d")
k_top_tokens = st.number_input(
    "Number of top tokens to show: ", min_value=0, step=1, value=3, format="%d")

submit_butt = st.button("Submit", type="primary")
if submit_butt:
    if not st.session_state['address_input']:
        st.warning("Please enter a valid address before submitting.")
    elif not is_valid_address(st.session_state['address_input'], wp2):
        st.warning("Invalid address. Try again.")
    else:
        st.success("Submission successful!")
        st.subheader(f"Address: {st.session_state['address_input']}")

        # Data calculation
        matrix, considered_address, edges = map_to_matrix(
            w2w, wp2, st.session_state['address_input'], periods)
        pageRank_values = cluster(matrix)
        total_balance = total_balance_in_USD(ethw, pageRank_values)

        # Cluster visualization
        st.subheader(f"Cluster Visualization")
        G = nx.DiGraph()
        G.add_nodes_from([address[:6] for address in pageRank_values])
        G.add_edges_from([(address_src[:6], address_dst[:6])
                         for (address_src, address_dst, count) in edges])

        node_sizes = [pageRank_values[address] *
                      1000 for address in pageRank_values]
        max_sizes = max(node_sizes)
        node_sizes = [size / max_sizes * 4000 for size in node_sizes]
        norm = plt.Normalize(min(node_sizes), max(node_sizes))
        node_colors = plt.cm.Pastel1(norm(node_sizes))
        node_labels = {
            address[:6]: f"{address[:6]}\n{pageRank_values[address] * 100:.2f}" for address in pageRank_values
        }

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        nx.draw(G,
                nx.spring_layout(G, k=3, seed=20),
                with_labels=False,
                node_size=node_sizes,
                node_color=node_colors,
                font_size=10,
                font_color="black",
                font_weight="bold",
                arrowsize=10)

        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, nx.spring_layout(
            G, k=3, seed=20), labels=node_labels, font_size=7, font_color="black")
        plt.title(f'Cluster balance: {total_balance}')
        st.pyplot(fig)

        # Top wallets and top tokens
        st.subheader(f"Top {k_top_wallets} Wallet Addresses By Distribution")
        st.dataframe(show_top_wallets(pageRank_values, k_top_wallets))
        st.subheader(f"Top {k_top_tokens} Tokens Involved")
        st.dataframe(show_top_tokens(w2sc, ethsc, st.session_state['address_input'],
                     pageRank_values, k_top_tokens, periods))
