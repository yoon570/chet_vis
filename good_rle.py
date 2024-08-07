from itertools import takewhile, islice, chain
from collections import Counter
from math import ceil
import numpy as np
from hybrid_approach_literal_escape import bse_encode, bse_decode
from test_alg import parallel_test_compression_ratio

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


def rle_encode(page, tolerance=0):  # change this to be 256
    compressed = list(page)
    unused = set(range(256)) - set(compressed)
    if not unused:
        c = Counter(page)
        least_used = min(c, key=lambda x: c[x])
        locations = []
        for idx, num in enumerate(page):
            if num == least_used:
                locations.append(idx)
        stride_escape_replacements = len(locations)
        compressed = [num for num in compressed if num != least_used]
        locations = [bin(location)[2:].zfill(12) for location in locations]
        locations = "".join(locations)
    else:
        least_used = min(unused)
        stride_escape_replacements = 0
        locations = ""

    stride_escape = least_used
    stride_escape_byte = least_used.to_bytes(length=1, byteorder="big")
    stride_escape_replacements_byte = stride_escape_replacements.to_bytes(
        length=1, byteorder="big"
    )
    min_bytes = ceil(len(locations) / 8)
    location_bytes = bytes(
        int("".join(num), 2) for num in batched(locations.ljust(8 * min_bytes, "0"), 8)
    )
    # okay now have everything just need to actually compress
    # this is the actual compression area
    # do rle if savings is above the tolerance. 1 byte escape, then specify length up to 256 bytes of zeros?
    # this can be it for now, work on more patterns over weekend. Just optimize rle and see what happens

    new_compressed = []
    idx = 0
    # maybe just focus on runs of zeros? Do other things even occur? Would save a whole byte.
    # only other thing I've seen is runs of FF, but only in certain dumps

    # cost would be 2 (length byte and escape). Maybe tolerance is just count based not savings based tho
    # easier
    while idx < len(compressed):
        if compressed[idx] == 0:
            count = 0
            while idx < len(compressed) and count < 64 and compressed[idx]==0:
                idx += 1
                count += 1
            if count >= tolerance:
                new_compressed.append(stride_escape)
                new_compressed.append(count)
            else:
                new_compressed.extend([0]*count)
        else:
            new_compressed.append(compressed[idx])
            idx += 1

    return (
        stride_escape_byte
        + stride_escape_replacements_byte
        + location_bytes
        + bytes(new_compressed)
    )


def rle_decode(compressed):
    stride_escape = compressed[0]
    stride_count = compressed[1]

    binary_form = np.unpackbits(np.frombuffer(compressed[2:], dtype=np.uint8))
    idx = 0
    locations = set()
    for _ in range(stride_count):
        locations.add(int("".join(map(str, binary_form[idx : idx + 12])), 2))
        idx += 12
    # now restore those bytes at the very end.
    idx += (8 - idx % 8) % 8
    page = np.packbits(binary_form[idx:]).tolist()

    # decompress right here:
    uncompressed = []
    idx = 0 
    while idx < len(page):
        if page[idx] == stride_escape:
            uncompressed.extend([0]*page[idx+1])
            idx += 2
        else:
            uncompressed.append(page[idx])
            idx += 1
    page = uncompressed

    # restore literal escapes
    assert stride_escape not in page

    for idx in sorted(locations):
        page.insert(idx, stride_escape)
    return bytes(page)

def bse_rle_encode(page, memory_blocks, tolerance=0):
    return bse_encode(rle_encode(page, tolerance), memory_blocks)

def bse_rle_decode(compressed):
    return rle_decode(bse_decode(compressed))

if __name__ == "__main__":
    # pages = read_file_in_chunks(f'downsampled_tests/declipseparsed_downscale_100')
    # print(list(pages[0]))
    # print(rle_encode([0]*20))
    benchmarks = [
        "declipseparsed",
        "parsec_splash2x.water_spatial5dump",
        "sparkbench_LogisticRegression5dump",
        "xalanparsed",
    ]
    for benchmark in benchmarks:
        pages = read_file_in_chunks(f"downsampled_tests/{benchmark}_rand1k")
        for mem_block_count in range(1, 4):
            for tolerance in [3]:
                results = parallel_test_compression_ratio(
                    pages,
                    bse_rle_encode,
                    bse_rle_decode,
                    [mem_block_count, tolerance],
                )
                with open(
                    f"new_stuff/{benchmark}.rle_bse_limit64", "a"
                ) as f:
                    f.write(f"tol: {tolerance}, mem blocks: {mem_block_count}, {results}\n")