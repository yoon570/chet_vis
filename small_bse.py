from itertools import islice, batched, takewhile
from collections import deque, defaultdict
import numpy as np
from byteobj import Byte, PageInfo
import pickle

def read_file_in_chunks(filename, chunk_size=4096):
    with open(filename, "rb") as file:
        chunk_iter = iter(lambda: file.read(chunk_size), b"")
        return list(takewhile(lambda chunk: chunk, chunk_iter))

def seq_lens_to_bytes(sub_buckets):
    lens = ""
    for bucket in sub_buckets:
        assert len(bucket) <= 63
        lens += bin(len(bucket))[2:].zfill(6)
    return bytes([int("".join(chunk), 2) for chunk in batched(lens, 8)])

def bytes_to_seq_lens(bytes):
    bits = "".join(map(str, np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))))
    counts = [int("".join(num), 2) for num in batched(bits, 6)]
    return counts

def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)

def bse_iteration(page, substitution):
    # different now
    util_coefs = {2: 1/3, 3: 1/2, 4: 3/5, 5: 2/3}
    # (-1 + util) * freq
    substitution = substitution.to_bytes()
    c = defaultdict(lambda: -1)
    for seq_len in range(2, 6):
        for seq in sliding_window(page, seq_len):
            c[seq] += util_coefs[len(seq)]
    best = max(c, key=lambda x:c[x])

    new_page = page.replace(bytes(best), substitution)

    if len(new_page) + len(best)+1 >= len(page):  # good for now. actually fix later
        return b"", page
    # 2 bits on average to encode length of sequence
    dict_entry = (len(best)-2, substitution, bytes(best))
    return dict_entry, new_page


def bse_encode(page, memory_blocks):
    subs = []
    compressed = page
    total_bits = memory_blocks * 64 * 8 - 24 # -24 for seq len bits

    unused = sorted(set(range(256)) - set(page), reverse=True)
    while len(subs) < 63:
        if not unused:
            break
        sub_byte = unused.pop()
        old = compressed
        sub, compressed = bse_iteration(compressed, sub_byte)
        if not sub or total_bits-(8+len(sub[2])*8) < 0:
            compressed = old
            break
        total_bits -= 8 + len(sub[2]) * 8 
        subs.append(sub)

    sub_buckets = [[] for _ in range(4)]
    for sub in subs:
        sub_buckets[sub[0]].append(sub[1:])

    seq_len_bytes = seq_lens_to_bytes(sub_buckets)
    subs_and_seqs = b""
    for bucket in sub_buckets:
        for sub, seq in bucket:
            subs_and_seqs += sub+seq
        
    dict_bytes = seq_len_bytes + subs_and_seqs
    assert len(dict_bytes) <= memory_blocks*64 
    return dict_bytes + bytes(compressed)


def bse_decode(compressed):
    
    counts, compressed = bytes_to_seq_lens(compressed[:3]), compressed[3:]

    length_groups = []
    for idx, count in enumerate(counts):
        entry_len = 1 + idx+2 # sub then idx+2
        length_groups.append(compressed[:entry_len*count])
        compressed = compressed[entry_len*count:]
    
    expansions = [[0]*5 for _ in range(256)]
    seq_lengths = [0] * 256

    for idx, group in enumerate(length_groups):
        entry_len = 1 + idx+2
        length = idx + 2
        # This is where the actual replacements are being generated
        for entry in batched(group, entry_len):
            sub = entry[0]
            seq = entry[1:]
            for idy, num in enumerate(seq):
                expansions[sub][idy] = num
            seq_lengths[sub] = length


    dict_entries = []
    for sub in range(256):
        length = seq_lengths[sub]
        if length:
            dict_entries.append((sub, expansions[sub][:length]))

    page = list(compressed[::-1])
    
    # Create a new byte_storage object
    byte_storage = []
    
    # This needs to be the ByteStorage after Huffman is applied here in the final version
    for num in page:
        byte_storage.append(Byte(num, 0, [], 0))

    output = []
    output_bs = []
    
    occur = {}

    while page:
        # Pop first element
        num = page.pop()
        curr_byte = byte_storage.pop()
        
        if seq_lengths[num]: # on expansion list
            
            if num not in occur:
                occur[num] = 1
            else:
                occur[num] += 1
                
            for idx in reversed(range(seq_lengths[num])):
                page.append(expansions[num][idx])
                
                new_byte = Byte(expansions[num][idx], curr_byte.stage1, curr_byte.stage2, curr_byte.stage3)
                if num not in curr_byte.stage2:
                    new_byte.stage2.append(num)
                byte_storage.append(new_byte)
        else:
            output.append(num)
            output_bs.append(curr_byte)
            
    pickle_page: PageInfo = PageInfo(dict_entries, occur)
    with open('page_data.pickl', 'wb') as pfile:
        pickle.dump(pickle_page, pfile)
            
    return bytes(output), output_bs

if __name__ == "__main__":
    benchmarks = [
        "declipseparsed",
        "parsec_splash2x.water_spatial5dump",
        "sparkbench_LogisticRegression5dump",
        "xalanparsed",
    ]

    for benchmark in benchmarks:
        pages = read_file_in_chunks(f"downsampled_tests/{benchmark}_rand1k")
        
        for page in pages:
            for subpage in batched(page, 4096):
                subpage = bytes(subpage)
                compressed = bse_encode(subpage, 3)
                decoded = bse_decode(compressed)
                print(len(compressed), decoded==subpage)