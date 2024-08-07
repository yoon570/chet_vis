import heapq
from collections import Counter
from itertools import count


class Node:
    def __init__(self, left=None, right=None, symbol=None):
        self.symbol = symbol
        self.left = left
        self.right = right


def create_huffman(page, k):
    h = list(page)
    c = Counter(h)
    most_frequent = c.most_common(k)
    remaining = (c - Counter(dict(most_frequent))).total()
    i = count()
    most_frequent = [
        (freq, next(i), Node(symbol=num)) for num, freq in most_frequent
    ] + [
        (remaining, next(i), Node(symbol=-1))
    ]  # escape char
    heapq.heapify(most_frequent)
    while len(most_frequent) > 1:
        f1, _, node1 = heapq.heappop(most_frequent)
        f2, _, node2 = heapq.heappop(most_frequent)
        heapq.heappush(most_frequent, (f1 + f2, next(i), Node(node1, node2)))
    tree = most_frequent[0][2]
    return tree


def find_codes(tree):
    bit_conversion = {}

    def dfs(root, path):
        if not root:
            return
        if root.symbol is not None:
            bit_conversion[root.symbol] = "".join(path)
        else:
            dfs(root.left, path + ["0"])
            dfs(root.right, path + ["1"])

    dfs(tree, [])
    return bit_conversion


def huffman_encode(page, k):
    tree = create_huffman(page, k)
    codes = find_codes(tree)
    code_bits = max(len(code) for code in codes.values())
    first_byte = bin(code_bits)[2:].zfill(8)  # code bits each
    symbols = [
        (bin(symbol)[2:].zfill(8), bin(len(code))[2:].zfill(code_bits), code)
        for symbol, code in codes.items()
        if symbol != -1
    ]
    second_byte = bin(len(symbols))[2:].zfill(8)  # number of codes
    after_symbols = bin(len(codes[-1]))[2:].zfill(code_bits) + codes[-1]
    message = []
    for char in page:
        if char in codes:
            message.append(codes[char])
        else:
            message.append(codes[-1] + bin(char)[2:].zfill(8))
    message = "".join(message)
    bit_count = bin(len(message))[2:].zfill(16)  # first 3rd and 4th bytes
    final_message = (
        first_byte
        + second_byte
        + bit_count
        + "".join(a + b + c for a, b, c in symbols)
        + after_symbols
        + message
    )
    final_message += "0" * ((8 - (len(final_message) % 8)) % 8)
    return int(final_message, 2).to_bytes(len(final_message) // 8, byteorder="big")


def huffman_decode(message_bytes):
    code_bits = message_bytes[0]
    code_count = message_bytes[1]
    bit_count = message_bytes[2] * 2**8 + message_bytes[3]
    binary_rep = "".join(bin(num)[2:].zfill(8) for num in message_bytes)
    codes = {}
    index = 4 * 8
    while len(codes) < code_count:
        symbol = int(binary_rep[index : index + 8], 2)
        index += 8
        code_length = int(binary_rep[index : index + code_bits], 2)
        index += code_bits
        code = binary_rep[index : index + code_length]
        codes[code] = symbol
        index += code_length
    escape_char_length = int(binary_rep[index : index + code_bits], 2)
    index += code_bits
    escape_code = binary_rep[index : index + escape_char_length]
    index += escape_char_length
    codes[escape_code] = -1

    binary_rep = binary_rep[index : index + bit_count]
    message = []
    while binary_rep:
        for code in codes:
            if binary_rep.startswith(code):
                if codes[code] == -1:
                    message.append(int(binary_rep[len(code) : len(code) + 8], 2))
                    binary_rep = binary_rep[len(code) + 8 :]
                    break
                message.append(codes[code])
                binary_rep = binary_rep[len(code) :]
                break
    return bytes(message)
