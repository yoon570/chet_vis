from itertools import takewhile, islice
from collections import defaultdict, Counter
from math import ceil
import numpy as np
from huffman import huffman_encode, huffman_decode

from test_alg import parallel_test_compression_ratio
# from test_alg import serial_test_compression_ratio

 

def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def read_file_in_chunks(filename, chunk_size=4096):
    with open(filename, "rb") as file:
        chunk_iter = iter(lambda: file.read(chunk_size), b"")
        return list(takewhile(lambda chunk: chunk, chunk_iter))


# units are bits in below functions
def uncompressed_size(sequence, freq):
    return freq * len(sequence) * 8


def dict_entry_size(sequence):
    # 3 bits for sequence length (2-8), 8 bits per byte of replaced sequence,
    # 8 bits for replacement byte
    return 3 + 8 * len(sequence) + 8


def compressed_size(sequence, freq):
    # each replaced instance is still 8 bits
    return dict_entry_size(sequence) + 8 * freq


def saving(sequence, freq):
    return uncompressed_size(sequence, freq) - compressed_size(sequence, freq)


def utility(sequence, freq, cost):
    return (saving(sequence, freq) - cost) / dict_entry_size(sequence)


# finds all non-self-overlapping byte-sequences of length n in page
def adjusted_sequence_frequencies(page, n, literal_escape):
    sequences = defaultdict(list)
    for i in range(len(page) - n + 1):
        current_seq = tuple(page[i : i + n])
        if i > 0 and page[i - 1] == literal_escape:
            continue
        if current_seq[-1] == literal_escape:
            continue
        if current_seq in sequences and sequences[current_seq][-1] + n > i:
            # overlaps
            continue
        sequences[current_seq].append(i)
    sequences = [(seq, len(freq)) for seq, freq in sequences.items()]
    return sequences


def get_best_util_sequence(page, remaining_bits, literal_escape, cost):
    sequences = []
    for n in range(2, 10):
        # don't want to exceed block limit
        if dict_entry_size([0] * n) <= remaining_bits:
            sequences.extend(adjusted_sequence_frequencies(page, n, literal_escape))

    # print(
    #     sorted(
    #         [(utility(*sequence, cost), sequence) for sequence in sequences],
    #         reverse=True,
    #     )[:10]
    # )

    if sequences:
        return True, max(sequences, key=lambda x: utility(*x, cost))
    # might not have any valid left
    return False, None


# gets string form of binary representing length. Always three bits
def sequence_length_to_binary(length):
    # 2 -> 000, ...,  8 -> 110
    return bin(length - 2)[2:].zfill(3)


def binary_to_sequence_length(binary):
    # 000 -> 2, ...,  110 -> 8
    return int(binary, 2) + 2


def byte_to_binary(byte):
    return bin(byte)[2:].zfill(8)


def replace_sequence(page, sequence, replacement, literal_escape):
    new_page = []
    idx = 0
    n = len(sequence)
    while idx < len(page):
        if tuple(page[idx : idx + n]) == sequence:
            new_page.append(replacement)
            idx += n
            continue
        # should skip over all relevant sequences of lit escapes
        if page[idx] == literal_escape:
            count = 0
            while page[idx + count] == literal_escape:
                count += 1
            new_page.extend(page[idx : idx + count + 1])
            idx += count + 1
        else:
            new_page.append(page[idx])
            idx += 1
    return new_page


def replace_byte(page, byte, replacement, literal_escape):
    new_page = []
    idx = 0
    if literal_escape in replacement:
        new_replacement = []
        index = 0
        while index < len(replacement):
            if replacement[index] == literal_escape and replacement[index + 1] == byte:
                new_replacement.append(byte)
                index += 1
            else:
                new_replacement.append(replacement[index])
            index += 1
        replacement = new_replacement
    while idx < len(page):
        if page[idx] == byte:
            new_page.extend(replacement)

        elif page[idx] == literal_escape and page[idx + 1] == byte:
            # effectively remove one lit escape
            new_page.append(byte)
            idx += 1
        else:
            new_page.append(page[idx])
        idx += 1
    return new_page


# returns page with least used byte escaped with literal escape byte, new
# "unused byte," and cost in bits
def create_unused(page, literal_escape):
    new_page = []
    c = Counter(page)
    c[literal_escape] = 4096  # make sure lit escape is never used
    least_used = min(c, key=lambda x: c[x])
    for num in page:
        if num == least_used:
            new_page.append(literal_escape)
        new_page.append(num)
    return new_page, least_used, c[least_used] * 8


# return bit string representing merge + new page after merge
# merge area stored in bit form, actual page always as bytes
def bse_iteration(page, remaining_bits, literal_escape):
    unused = set(range(256)) - set(page) - {literal_escape}
    # now strategy is to create space when needed with literal escape
    # if no unused, make one and see if it's still worth
    # so least used char will get escape prepended, then those are
    # essentially ignored later during expansion
    original_page = page
    if not unused:
        page, least_used, cost = create_unused(page, literal_escape)
        unused = {least_used}
    else:
        least_used = None
        cost = 0

    substitution = min(unused) # make it deterministic for analysis in dumps
    # filter out sequences that are too large to fit
    result, best = get_best_util_sequence(page, remaining_bits, literal_escape, cost)

    if not result:
        return "", original_page
    best_seq, best_freq = best

    predicted_saving = (
        saving(best_seq, best_freq) - cost
    )  # cost factors the escaping in properly

    if predicted_saving <= 0:
        return "", original_page

    sequence_length_bits = sequence_length_to_binary(len(best_seq))
    sequence_bits = "".join(map(byte_to_binary, best_seq))
    substitution_bits = byte_to_binary(substitution)
    dict_entry = sequence_length_bits + sequence_bits + substitution_bits
    replaced_page = replace_sequence(page, best_seq, substitution, literal_escape)

    # just a check to make sure
    actual_saving = len(original_page) * 8 - (len(replaced_page) * 8 + len(dict_entry))
    assert actual_saving == predicted_saving

    return dict_entry, replaced_page


# penalty is the impact from static table in BYTES.
def bse_encode(page, memory_blocks, penalty=0):
    # page = csc_encode(page, 256)
    dict_entries = []
    compressed = list(page)
    # need to force a unused bit right here.
    unused = set(range(256)) - set(compressed)
    # first byte is entry count
    # then next byte is literal escape
    # then count of 12 bit literal escape locations
    # then those bits, then dict entries
    if not unused:
        # need to create one
        c = Counter(page)
        least_used = min(c, key=lambda x: c[x])
        locations = []
        for idx, num in enumerate(page):
            if num == least_used:
                locations.append(idx)
        literal_escape_replacements = len(locations)
        # get rid of instances of least used byte here
        compressed = [num for num in compressed if num != least_used]
        locations = [bin(location)[2:].zfill(13) for location in locations]
        locations = "".join(locations)
    else:
        # already zero use candidate
        least_used = min(unused)
        literal_escape_replacements = 0
        locations = ""
    literal_escape = least_used
    literal_escape_byte = least_used.to_bytes(length=1, byteorder="big")
    literal_escape_replacements_byte = literal_escape_replacements.to_bytes(
        length=1, byteorder="big"
    )
    # 64 bytes per block, 8 bits per byte, 8 bits used for entry count
    # 8 bits for literal escape, 12*location entries
    remaining_bits = memory_blocks * 64 * 8 - 8 - 8 - len(locations) - penalty * 8
    # should stop when progress cannot be made or memory blocks used up.
    while len(dict_entries) < 255:
        old_size = len(compressed)
        dict_entry, compressed = bse_iteration(
            compressed, remaining_bits, literal_escape
        )
        if len(compressed) > old_size:
            print("got bigger whoops")
            raise Exception
        if not dict_entry:
            break
        remaining_bits -= len(dict_entry)
        dict_entries.append(dict_entry)

    entry_count = len(dict_entries).to_bytes(length=1, byteorder="big")
    # pack dict_entries into bytes.
    dict_entries = "".join(dict_entries)
    # insert locations right here
    dict_entries = locations + dict_entries
    min_bytes = ceil(len(dict_entries) / 8)
    dict_bytes = bytes(
        int("".join(num), 2)
        for num in batched(dict_entries.ljust(8 * min_bytes, "0"), 8)
    )
    just_bse = entry_count + literal_escape_byte + literal_escape_replacements_byte + dict_bytes + bytes(compressed)

    # return just_bse
    huffman_encoded =  huffman_encode(just_bse, 15)
    # can run decoder on a single byte to figure out what to highlight
    # can just pass those to yoon for now? eventually embed them into the dump?
    if len(just_bse) < len(huffman_encoded):
        return bytes([0])+just_bse
    return bytes([1])+huffman_encoded

def bse_decode(compressed):
    huffman, compressed = compressed[0], compressed[1:]
    if huffman:
        compressed = huffman_decode(compressed)

    dict_entries = []
    dict_entry_count = compressed[0]
    literal_escape = compressed[1]
    literal_count = compressed[2]

    binary_form = np.unpackbits(np.frombuffer(compressed[3:], dtype=np.uint8))
    idx = 0
    locations = set()
    for _ in range(literal_count):
        locations.add(int("".join(map(str, binary_form[idx : idx + 13])), 2))
        idx += 13
    # now restore those bytes at the very end.
    for _ in range(dict_entry_count):
        seq_len = binary_to_sequence_length(
            "".join(map(str, binary_form[idx : idx + 3]))
        )
        idx += 3
        sequence = tuple(
            int("".join(map(str, batch)), 2)
            for batch in batched(binary_form[idx : idx + 8 * seq_len], 8)
        )
        idx += 8 * seq_len
        replacement = int("".join(map(str, binary_form[idx : idx + 8])), 2)
        idx += 8
        dict_entries.append((sequence, replacement))

    # fix padding (skip to next mult of 8)
    idx += (8 - idx % 8) % 8
    page = np.packbits(binary_form[idx:]).tolist()
    for replacement, byte in reversed(dict_entries):
        page = replace_byte(page, byte, replacement, literal_escape)

    assert literal_escape not in page
    for idx in sorted(locations):
        page.insert(idx, literal_escape)
    return bytes(page)

import glob
if __name__ == "__main__":
    directory_path = 'downsampled_tests/*'
    file_paths = glob.glob(directory_path)
    for benchmark in file_paths:
        pages = read_file_in_chunks(benchmark)
        for mem_block_count in range(1, 4):
            results = parallel_test_compression_ratio(
                pages,
                bse_encode,
                bse_decode,
                [mem_block_count],
            )
            # with open(
            #     f"new_downsampled_data/{benchmark}.lit_esc_bse_dyna_huff", "a"
            # ) as f:
            #     f.write(f"mem blocks: {mem_block_count}, {results}\n")