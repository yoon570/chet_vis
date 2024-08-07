from itertools import takewhile, islice, chain
from collections import Counter
from math import ceil
import numpy as np
from test_alg import parallel_test_compression_ratio_save, parallel_test_compression_ratio, serial_test_compression_ratio
from huffman import huffman_encode, huffman_decode
# from csc import compress_block, decompress_block
from good_csc import compress_block, decompress_block
# from good_csc_zero import compress_block, decompress_block
# from hybrid_approach_literal_escape import bse_encode, bse_decode
from bse_static_dynamic_unified import bse_encode, bse_decode, gen_static_table
from byteobj import Byte

def create_labeled_memory_dump(
    pages, filename, tag_char="@", magic_bytes=b"\xDE\xAD\xBE\xEF"
):
    """
    Writes a labeled memory dump to a file with dynamic multi-block headers.

    :param pages: A list of tuples, each tuple contains byte sequences (page content),
    additional header string, and number of bytes to separate.
    :param filename: The name of the output file.
    :param tag_char: The character to use for the tag line (default is '@').
    :param magic_bytes: The sequence of magic bytes to append at the end of each header
    block and the page content.
    """
    with open(filename, "wb") as f:
        for page_number, (page_content, sizes) in enumerate(pages, start=1):
            # Construct the full header with dynamic content
            # should provide the labeled version and unlabeled version for data
            # collection reasons.
            # count instances of the first byte in the compressed verion?
            # can count instances of "<<<" to figure out how much to adjust?
            # original, one_byte_encoding, zero_byte_encoding
            full_header_info = (
                f"{tag_char * 10} Page: {page_number} Sizes: {sizes}".encode("ascii")
            )
            # Calculate needed blocks and padding for the header
            full_header_length = len(full_header_info) + len(magic_bytes)
            full_header_blocks = (full_header_length + 63) // 64
            full_header_padded_length = full_header_blocks * 64
            header = (
                full_header_info.ljust(
                    full_header_padded_length - len(magic_bytes), b"\x00"
                )
                + magic_bytes
            )

            # Write the full header
            f.write(header)

            # Separate the initial part of the page
            remaining_part_length_with_magic = len(page_content) + len(magic_bytes)
            padding_length_remaining = (
                64 - (remaining_part_length_with_magic % 64)
            ) % 64
            remaining_padding = b"\x00" * padding_length_remaining

            # Assemble and write the initial and remaining part of the page
            f.write(page_content + magic_bytes + remaining_padding)


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


def csc_encode(page, max_block=256, tolerance=0, label=False):  # change this to be 256
    # break it up into max size blocks
    # with open('correct.data', 'wb') as f:
    # f.write(bytes(page))
    compressed = list(page)
    # need to force a unused bit right here.
    unused = set(range(256)) - set(compressed)
    # first byte is entry count
    # then next byte is constant_stride escape
    # then count of 12 bit constant_stride escape locations
    # then those bits, then dict entries
    if not unused:
        # need to create one
        c = Counter(page)
        least_used = min(c, key=lambda x: c[x])
        locations = []
        for idx, num in enumerate(page):
            if num == least_used:
                locations.append(idx)
        # print(locations)
        stride_escape_replacements = len(locations)
        # get rid of them now
        compressed = [num for num in compressed if num != least_used]
        locations = [bin(location)[2:].zfill(13) for location in locations]
        locations = "".join(locations)
    else:
        # already zero use candidate
        least_used = unused.pop()
        stride_escape_replacements = 0
        locations = ""

    stride_escape = least_used
    stride_escape_byte = least_used.to_bytes(length=1, byteorder="big")
    stride_escape_replacements_byte = stride_escape_replacements.to_bytes(
        length=1, byteorder="big"
    )
    # 64 bytes per block, 8 bits per byte, 8 bits used for entry count
    # 8 bits for literal escape, 12*location entries
    # need to remember to actually remove those least used ones after storing locations
    min_bytes = ceil(len(locations) / 8)
    location_bytes = bytes(
        int("".join(num), 2) for num in batched(locations.ljust(8 * min_bytes, "0"), 8)
    )
    # okay now have everything just need to actually compress
    blocks = list(batched(compressed, max_block))
    compressed_blocks = [
        compress_block(list(block), stride_escape, tolerance=tolerance, label_mode=label) for block in blocks
    ]
    compressed = list(chain(*compressed_blocks))

    return (
        stride_escape_byte
        + stride_escape_replacements_byte
        + location_bytes
        + bytes(compressed)
    )

def csc_decode(compressed, post_csc_bs):
    stride_escape = compressed[0]
    stride_count = compressed[1]

    binary_form = np.unpackbits(np.frombuffer(compressed[2:], dtype=np.uint8))
    idx = 0
    locations = set()
    for _ in range(stride_count):
        locations.add(int("".join(map(str, binary_form[idx : idx + 13])), 2))
        idx += 13
    # now restore those bytes at the very end.
    idx += (8 - idx % 8) % 8
    page = np.packbits(binary_form[idx:]).tolist()
    
    post_csc_bs = post_csc_bs[-len(page):]
    
    page_pair = decompress_block(page, stride_escape, post_csc_bs)
    page = page_pair[0]
    page_bs = page_pair[1]
    # restore literal escapes
    assert stride_escape not in page

    
    # reinserting strided escape chararacters at the locations, sorted, inside the page & ByteStorage
    for idx in sorted(locations):
        page.insert(idx, stride_escape)
        page_bs.insert(idx, Byte(stride_escape, 1, [], 0))
    return bytes(page), page_bs


def csc_label(page):
    compressed_labeled = csc_encode(page, max_block=256, label=True)
    compressed = csc_encode(page, max_block=256)
    stride_escape = compressed[0]
    csc_instances = compressed.count(stride_escape) - 1
    sizes = (
        len(compressed),
        len(compressed) - csc_instances,
        len(compressed) - 2 * csc_instances,
    )

    return (bytes(compressed_labeled), sizes)


def hybrid_huffman_encode(page, memory_blocks, k):
    compressed = csc_encode(page, memory_blocks)
    return huffman_encode(compressed, k)


def hybrid_huffman_decode(compressed):
    uncompressed = huffman_decode(compressed)
    return csc_decode(uncompressed)

# def bse_csc_encode(page, memory_blocks, static_table, tolerance=0):
#     return bse_encode(csc_encode(rle_encode(page, 3), 256, tolerance), memory_blocks, static_table)

# def bse_csc_decode(compressed, static_table):
#     return rle_decode(csc_decode(bse_decode(compressed, static_table)))

# def bse_csc_encode(page, memory_blocks, tolerance=0):
#     return bse_encode(csc_encode(page, 256, tolerance), memory_blocks)

# def bse_csc_decode(compressed):
#     return csc_decode(bse_decode(compressed))
def bse_csc_encode(page, memory_blocks, static_table, tolerance=0):
    return bse_encode(csc_encode(page, 256, tolerance), memory_blocks, static_table)

def bse_csc_decode(compressed, static_table):
    return csc_decode(bse_decode(compressed, static_table))


huge_arr = []
# try to pickle post_csc stuff for bse static table testing
# def bse_csc_encode(page, memory_blocks, tolerance=0):
#     compressed_out = csc_encode(rle_encode(page, 3), 256, tolerance)
#     huge_arr.append(compressed_out)
#     return compressed_out
# def bse_csc_decode(compressed):
#     return rle_decode(csc_decode(compressed))
# def bse_csc_encode(page, memory_blocks, tolerance=0):
#     compressed_out = csc_encode(page, 256, tolerance)
#     huge_arr.append(compressed_out)
#     return compressed_out
# def bse_csc_decode(compressed):
#     return csc_decode(compressed)



# def bse_csc_encode(page, memory_blocks, static_table, tolerance=0):
#     return bse_encode(csc_encode(page, 256, tolerance), memory_blocks, static_table)

# def bse_csc_decode(compressed, static_table):
#     return csc_decode(bse_decode(compressed, static_table))


# maybe for this test, do rle then good_csc
from csc import update_settings
from itertools import product

from good_rle import rle_encode, rle_decode

from good_csc import report_pattern_savings
import pickle
import glob
if __name__ == "__main__":
    # pages = read_file_in_chunks(f'downsampled_tests/declipseparsed_downscale_100')
    static_table = gen_static_table()
    # benchmarks = [
    #     "declipseparsed",
    #     "parsec_splash2x.water_spatial5dump",
    #     "sparkbench_LogisticRegression5dump",
    #     "xalanparsed",
    # ]
    directory_path = 'well_rounded_tests/*'
    file_paths = glob.glob(directory_path)
    for benchmark in file_paths:
        pages = read_file_in_chunks(benchmark)
        for mem_block_count in range(1, 4):
        # for mem_block_count in range(1, 2):
            # for tolerance in [0, 5, 10, 20]:
            for tolerance in [10]: # this is pretty good
                results = parallel_test_compression_ratio(
                # results = serial_test_compression_ratio(
                    pages, # truncate for now
                    bse_csc_encode,
                    bse_csc_decode,
                    # [mem_block_count, tolerance],
                    [mem_block_count, static_table, tolerance],
                    [static_table]
                )

                with open(
                    f"well_rounded_results/{benchmark}.big_5_patterns_unified_bse_tol_10_1k_pages", "a"
                ) as f:
                    # f.write(f"tol: {tolerance}, mem blocks: {mem_block_count}, pattern savings: {report_pattern_savings()}, {results}\n")
                    f.write(f"tol: {tolerance}, mem blocks: {mem_block_count}, {results}\n")
            # with open(f"{benchmark}_after_csc_no_rle.pkl", 'wb') as f:
                # pickle.dump(huge_arr, f)
                print()
                print(results, benchmark)
                print()
            huge_arr.clear()
   