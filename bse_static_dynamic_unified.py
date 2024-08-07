
# modifications coming:
# util cutoff, dict length instead of entry count first byte
# go to 5 byte sequences. it's time.
# adjustments to the dictionary. maybe

from itertools import takewhile, islice
from collections import defaultdict, Counter
from math import ceil
import numpy as np
from huffman import huffman_encode, huffman_decode

from test_alg import parallel_test_compression_ratio, serial_test_compression_ratio

# a little bit of degradation actually when making dict smaller for some. Definitely tuning needed
# tweaking should go after csc is fixed though because it can change what's present a lot

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


def dict_entry_size(sequence, static_table):
    # 1 bit for static or dynamic
    # if static, TABLE_SIZE_BITS + 8 for replacement
    # if dynamic, this stuff:
    # 2 bits for sequence length (2-5), 8 bits per byte of replaced sequence,
    # 8 bits for replacement byte
    if sequence in static_table:
        return 1 + TABLE_SIZE_BITS + 8
    return 1 + 2 + 8 * len(sequence) + 8


def compressed_size(sequence, freq, static_table):
    # each replaced instance is still 8 bits
    return dict_entry_size(sequence, static_table) + 8 * freq


def saving(sequence, freq, static_table):
    return uncompressed_size(sequence, freq) - compressed_size(sequence, freq, static_table)


def utility(sequence, freq, cost, static_table):
    return (saving(sequence, freq, static_table) - cost) / dict_entry_size(sequence, static_table)


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


def get_best_util_sequence(page, remaining_bits, literal_escape, cost, static_table):
    sequences = []
    for n in range(2, 6):
        # don't want to exceed block limit
        # if dict_entry_size([0] * n, static_table) <= remaining_bits:
        # ahh this is the issue. could be dynamic too
        if dict_entry_size([0] * n, static_table) <= remaining_bits:
            sequences.extend(adjusted_sequence_frequencies(page, n, literal_escape))

    sequences_small_enough = list(filter(lambda x: dict_entry_size(x[0], static_table) <= remaining_bits, sequences))
    if sequences_small_enough:
        return True, max(sequences_small_enough, key=lambda x: utility(*x, cost, static_table))
    # might not have any valid left
    return False, None


# gets string form of binary representing length. Always three bits
def sequence_length_to_binary(length):
    # 2 -> 00, ...,  5 -> 11
    return bin(length - 2)[2:].zfill(2)


def binary_to_sequence_length(binary):
    # 00 -> 2, ...,  11 -> 5
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
def bse_iteration(page, remaining_bits, literal_escape, static_table):
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
    result, best = get_best_util_sequence(page, remaining_bits, literal_escape, cost, static_table)

    if not result:
        return "", original_page
    best_seq, best_freq = best

    predicted_saving = (
        saving(best_seq, best_freq, static_table) - cost
    )  # cost factors the escaping in properly

    static_table.increment_savings(best_seq, predicted_saving)

    if predicted_saving <= 0:
        return "", original_page

    substitution_bits = byte_to_binary(substitution)
    if best_seq in static_table:
        static_index_bits = bin(static_table.index(best_seq))[2:].zfill(TABLE_SIZE_BITS)
        dict_entry = '1' + static_index_bits + substitution_bits
    else:
        sequence_length_bits = sequence_length_to_binary(len(best_seq))
        sequence_bits = "".join(map(byte_to_binary, best_seq))
        dict_entry = '0' + sequence_length_bits + sequence_bits + substitution_bits

    replaced_page = replace_sequence(page, best_seq, substitution, literal_escape)

    # just a check to make sure
    actual_saving = len(original_page) * 8 - (len(replaced_page) * 8 + len(dict_entry))
    assert actual_saving == predicted_saving

    return dict_entry, replaced_page


# penalty is the impact from static table in BYTES.
def bse_encode(page, memory_blocks, static_table, huffman=False):
    # page = csc_encode(page, 256)
    dict_entries = []
    compressed = list(page)
    # need to force a unused bit right here.
    unused = set(range(256)) - set(compressed)
    # first byte is entry count
    # then next byte is literal escape
    # then count of 13 bit literal escape locations. 13 in case last step expanded
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
    # 64 bytes per block, 8 bits per byte, 8 bits used for entry bytes (0-255)
    # 8 bits for literal escape, 8 bits for literal escape replacements, 12*location entries
    # then need to round down to a multiple of 8
    # idx += (8 - idx % 8) % 8
    # should work robustly now
    max_bytes = 64*memory_blocks
    max_bytes = min(max_bytes, 255) # can only go up to 255 for dict entry byte length
    remaining_bits = max_bytes*8  -8 -8 -8 - len(locations)
    remaining_bits -= remaining_bits%8

    # should stop when progress cannot be made or memory blocks used up.
    # while len(dict_entries) < 255:
    while True:
        old_size = len(compressed)
        dict_entry, compressed = bse_iteration(
            compressed, remaining_bits, literal_escape, static_table
        )
        if len(compressed) > old_size:
            print("got bigger whoops")
            raise Exception
        if not dict_entry:
            break
        remaining_bits -= len(dict_entry)
        dict_entries.append(dict_entry)

    # entry_count = len(dict_entries).to_bytes(length=1, byteorder="big")
    # pack dict_entries into bytes.
    dict_entries = "".join(dict_entries)
    # insert locations right here
    dict_entries = locations + dict_entries
    min_bytes = ceil(len(dict_entries) / 8)
    dict_bytes = bytes(
        int("".join(num), 2)
        for num in batched(dict_entries.ljust(8 * min_bytes, "0"), 8)
    )
    entry_bytes = len(dict_bytes).to_bytes(length=1, byteorder="big")
    just_bse = entry_bytes + literal_escape_byte + literal_escape_replacements_byte + dict_bytes + bytes(compressed)

    if not huffman:
        return just_bse

    huffman_encoded =  huffman_encode(just_bse, 15)
    # can run decoder on a single byte to figure out what to highlight
    # can just pass those to yoon for now? eventually embed them into the dump?
    if len(just_bse) < len(huffman_encoded):
        return bytes([0])+just_bse
    return bytes([1])+huffman_encoded

def bse_decode(compressed, static_table, huffman=False):
    if huffman:
        huffman, compressed = compressed[0], compressed[1:]
        if huffman:
            compressed = huffman_decode(compressed)

    dict_entries = []
    dict_entry_bytes = compressed[0]
    literal_escape = compressed[1]
    literal_count = compressed[2]

    binary_form = np.unpackbits(np.frombuffer(compressed[3:3+dict_entry_bytes], dtype=np.uint8))
    idx = 0
    locations = set()
    for _ in range(literal_count):
        locations.add(int("".join(map(str, binary_form[idx : idx + 13])), 2))
        idx += 13
    # now restore those bytes at the very end.
    while idx + 8 < len(binary_form): # just keep going until you run out of space
        if binary_form[idx]: # static
            idx += 1
            sequence = static_table[
                int("".join(map(str, binary_form[idx : idx + TABLE_SIZE_BITS])), 2)
            ]
            idx += TABLE_SIZE_BITS
            replacement = int("".join(map(str, binary_form[idx : idx + 8])), 2)
            idx += 8
            dict_entries.append((sequence, replacement))
        else:
            idx += 1
            seq_len = binary_to_sequence_length(
                "".join(map(str, binary_form[idx : idx + 2]))
            )
            idx += 2
            sequence = tuple(
                int("".join(map(str, batch)), 2)
                for batch in batched(binary_form[idx : idx + 8 * seq_len], 8)
            )
            idx += 8 * seq_len
            replacement = int("".join(map(str, binary_form[idx : idx + 8])), 2)
            idx += 8
            dict_entries.append((sequence, replacement))

    # fix padding (skip to next mult of 8)
    # idx += (8 - idx % 8) % 8
    page = list(compressed[3+dict_entry_bytes:])
    # page = np.packbits(binary_form[idx:]).tolist()
    for replacement, byte in reversed(dict_entries):
        page = replace_byte(page, byte, replacement, literal_escape)

    assert literal_escape not in page
    for idx in sorted(locations):
        page.insert(idx, literal_escape)
    return bytes(page)

from collections import defaultdict, Counter

def zero_generator():
    return 0
class FastStatic:
    def __init__(self, static_table):
        self.static_table = static_table
        self.static_lookup = {tup:idx for idx, tup in enumerate(static_table)}
        self.savings = defaultdict(zero_generator)
    def index(self, item):
        return self.static_lookup[item]
    def __getitem__(self, key):
        return self.static_table[key]
    def __contains__(self, item):
        return tuple(item) in self.static_lookup
    def increment_savings(self, item, saving):
        self.savings[item] += saving
    def report_savings(self):
        top_sequences = Counter(self.savings).most_common(50)
        self.savings.clear()
        return top_sequences

# TABLE_SIZE_BITS = 9 # log_2 of size
TABLE_SIZE_BITS = 10 # log_2 of size
# can do a lot of workshopping here. Maybe low bytes?
# maybe it's not that worth after we put csc in though
# since it was mostly targeting stuff that csc can do now

# get this table really small. gather the heavies

# size and contents needs a lot of optimization
# can do an interesting global op machine learning style
def gen_static_table():
    static_table = set()
    for num in range(256):
    # for num in range(127):
        static_table.add((num, num, num, num, num))
        static_table.add((num, num, num, num))
        static_table.add((num, num, num))
        # static_table.add((num, num))
        static_table.add((0, num, 0))

    assert (len(static_table)-1).bit_length() == TABLE_SIZE_BITS
    static_table = sorted(static_table)
    static_table = FastStatic(static_table)
    return static_table

import glob
import pickle
if __name__ == "__main__":
    static_table = gen_static_table()
    directory_path = 'well_rounded_tests/*'
    file_paths = glob.glob(directory_path)
    for benchmark in sorted(file_paths):
        # with open(f"{benchmark}", 'rb') as f:
            # pages = pickle.load(f)
        pages = read_file_in_chunks(benchmark)
        for mem_block_count in range(1, 5):
        # for mem_block_count in range(4, 5):
            results = parallel_test_compression_ratio(
            # results = serial_test_compression_ratio(
                pages,
                bse_encode,
                bse_decode,
                [mem_block_count, static_table],
                [static_table]
            )
            print(static_table.report_savings(), benchmark, results)
            # with open(
            #     f"new_stuff/{benchmark}.static_dynamic_unified", "a"
            # ) as f:
            #     f.write(f"mem blocks: {mem_block_count}, {results}\n")