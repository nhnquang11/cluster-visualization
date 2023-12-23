import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# wp2 = pd.read_csv('data/wallet_profile_2.csv')
# w2w = pd.read_csv('data/wallet_to_wallet.csv')
# ethw = pd.read_csv('data/eth_wallets.csv')
# ethsc = pd.read_csv('data/ETH_smart_contract.csv')
# w2sc = pd.read_csv('data/w2sc.csv')


def map_to_matrix(w2w, wp2, address: str, period: float = 1):
    num_blocks = int(period * 24 * 60 * 60 // 12)
    considered_blocks = w2w.tail(num_blocks)
    considered_addresses = [address]
    queue = [address]
    edges = []

    while len(queue) > 0:
        considered_address = queue.pop(0)
        receivers = considered_blocks[considered_blocks['from_address']
                                      == considered_address]['to_address'].value_counts().to_dict()
        for receiver, num_interactions in receivers.items():
            label = wp2[wp2['address'] == receiver]['label'].values
            if len(label) > 0 and label == 0:
                edges.append((considered_address, receiver, num_interactions))
                if receiver not in considered_addresses:
                    considered_addresses.append(receiver)
                    queue.append(receiver)

        senders = considered_blocks[considered_blocks['to_address'] ==
                                    considered_address]['from_address'].value_counts().to_dict()
        for sender, num_interactions in senders.items():
            label = wp2[wp2['address'] == sender]['label'].values
            if len(label) > 0 and label == 0:
                edges.append((sender, considered_address, num_interactions))
                if sender not in considered_addresses:
                    considered_addresses.append(sender)
                    queue.append(sender)

    matrix = pd.DataFrame(0, index=considered_addresses,
                          columns=considered_addresses)
    for sender, receiver, num_interactions in edges:
        matrix.loc[sender, receiver] = num_interactions

    return matrix, considered_addresses, edges


def cluster(matrix, d=0.85, max_iters=10000, tolerance=1e-20):
    n = len(matrix)
    sum_by_row = matrix.sum(axis=1)
    sum_by_row[sum_by_row == 0] = n
    transition_matrix = matrix.div(sum_by_row, axis=0)
    transition_matrix[transition_matrix.sum(axis=1) == 0] = 1 / n
    transition_matrix = transition_matrix.T
    pageRank_values = pd.Series(1 / n, index=matrix.index)
    progess_bar = tqdm(range(max_iters))
    for _ in progess_bar:
        prev_pageRank_values = pageRank_values.copy()
        pageRank_values = (1 - d) / n + d * \
            transition_matrix.dot(pageRank_values)
        diff = (pageRank_values - prev_pageRank_values).abs()
        progess_bar.set_postfix_str(
            f'Max difference {diff.max()}\tSum difference {diff.sum()}')
        if ((pageRank_values - prev_pageRank_values).abs() < tolerance).all():
            break

    return pageRank_values.sort_values(ascending=False).to_dict()


def show_top_wallets(pageRank_values, n=10):
    return pd.DataFrame.from_dict(pageRank_values, orient='index', columns=['PageRank']).head(n).reset_index().rename(columns={'index': 'Wallet address'})


def show_top_tokens(w2sc, ethsc, address, pageRank_values, n=10, period=1):
    num_blocks = int(period * 24 * 60 * 60 // 12)
    considered_blocks = w2sc.tail(num_blocks)

    top_tokens = considered_blocks[considered_blocks['from_address']
                                   == address]['to_address'].value_counts()
    for addr in pageRank_values:
        if addr != address:
            address_info = considered_blocks[considered_blocks['from_address']
                                             == addr]['to_address'].value_counts()
            top_tokens = top_tokens.add(
                address_info, fill_value=0).sort_values(ascending=False)

    top_tokens = top_tokens.head(n)

    top_tokens_info = {'Token address': top_tokens.index,
                       'Project': [], 'Category': []}

    for token in top_tokens_info['Token address']:
        top_tokens_info['Project'].append(
            ethsc[ethsc['contract_address'] == token]['project'].values[0])
        top_tokens_info['Category'].append(
            ethsc[ethsc['contract_address'] == token]['category'].values[0])

    return pd.DataFrame(top_tokens_info)


def total_balance_in_USD(ethw, pageRank_values):
    return ethw[ethw['address'].isin(pageRank_values.keys())]['balanceInUSD'].sum()


def get_random_address(wp2):
    return random.choice(wp2['address'])


def is_valid_address(address, wp2):
    return address in wp2['address'].values
