from itertools import takewhile
from copy import deepcopy
import numpy as np


def split_list(lst, val):
    # can't naively split because the byte after can be necessary
    groups = []
    idx = 0
    group = []
    while idx < len(lst):
        if lst[idx] != val:
            group.append(lst[idx])
        else:
            groups.append(group[:])
            group = lst[idx + 1 : idx + 3]  # pre-add next two
            idx += 2
        idx += 1
    if group:
        groups.append(group)
    return groups


def read_file_in_chunks(filename, chunk_size=4096):
    with open(filename, "rb") as file:
        chunk_iter = iter(lambda: file.read(chunk_size), b"")
        return list(takewhile(lambda chunk: chunk, chunk_iter))


# one-paper version
# settings = [3, 3, 4] # length bits, stride bits, repeat bits
# maybe one that fits in a single byte. need to test to see if this is a good setting

# start with single bit that represents means the sequence is all zeros
# maybe the repeat bits goes down? Getting crazy so go back to 2 byte version
# settings = [2, 3, 3]  # length bits, stride bits, repeat bits

# can dynamically play with any settings now.
# can figure out which of the counts is getting used the least,
# most like the the stride bits, so go down to 4

# should add to 1 under a multiple of 8

# just test all versions stars and bars that add to 7?
settings = [5, 3, 7] # length bits, stride bits, repeat bits

len_bits, stride_bits, repeat_bits = settings

def update_settings(new_settings):
    global settings, len_bits, stride_bits, repeat_bits
    settings = new_settings
    len_bits, stride_bits, repeat_bits = settings


# i think I see the problem. it shouldn't include it in the cost?
# but should consider it for selection?
# so am I accidentally double penalizing it?
def compressed_size(seq, stride, count):
    # 10 for stride bits and stuff 8 bits for escape.
    if seq == [0]*len(seq):
        return sum(settings)+1 + 8
    return sum(settings)+1 + 8 + len(seq)*8


def uncompressed_size(seq, stride, count):
    return len(seq) * (count+1) * 8


def savings(args):
    return uncompressed_size(*args) - compressed_size(*args)


def affected_indices(idx, seq_len, stride, count):
    seq_len = len(seq_len)
    idx += stride + seq_len
    indices = []
    for _ in range(count):
        for num in range(seq_len):
            indices.append(idx + num)
        idx += seq_len + stride
    return indices


def nums_to_bytes(best_option):
    # packs into byte
    seq, stride, repeat = best_option
    length = len(seq)
    length = bin(length-1)[2:].zfill(len_bits)
    stride = bin(stride)[2:].zfill(stride_bits)
    repeat = bin(repeat - 1)[2:].zfill(repeat_bits)
    binary_arr = [1 if num == "1" else 0 for num in length + stride + repeat]
    zero_seq = int(seq == [0]*len(seq))
    binary_arr.append(zero_seq)
    return np.packbits(binary_arr).tolist()


def byte_to_nums(bytes):
    binary_form = "".join(
        ["1" if num else "0" for num in np.unpackbits(np.array(bytes, dtype=np.uint8))]
    )
    length, stride, repeat = (
        binary_form[:len_bits],
        binary_form[len_bits:-repeat_bits-1],
        binary_form[-repeat_bits-1:-1],
    )
    # print(length, stride, )
    length = int(length, base=2) + 1
    stride = int(stride, base=2)
    repeat = int(repeat, base=2) + 1
    zero_seq = int(binary_form[-1])
    return length, stride, repeat, zero_seq


def nums_to_bytes_ascii(escape_nums):
    nums = (savings(escape_nums) // 8,) + escape_nums
    return list(f"<<<{nums}>>>".replace(" ", "").encode("ascii"))


# tolerance of 10 seems to be good (just empirically)
# actually higher is better so that it gives it a chance to be useful in combination
#  with 1 mem block bse
# 100 seems to be best
def compress_block(block, stride_escape, tolerance=0, label_mode=False):
    block = deepcopy(block)
    idx = 0
    while idx < len(block):
        options = []
        for seq_len in range(1, 1 + 2**len_bits):
            pattern = block[idx : seq_len + idx]
            for stride in range(2**stride_bits):
                count = 0  # first doesn't count. it's for free
                curr_idx = idx + len(pattern) + stride
                if curr_idx > len(block):
                    break
                # can repeat bits start at 2?
                while (
                    pattern == block[curr_idx : curr_idx + len(pattern)]
                    and count < 2**repeat_bits
                ):
                    count += 1
                    curr_idx += len(pattern) + stride
                if count >= 1:
                    # print((pattern, stride, count))
                    options.append((pattern, stride, count))
# means zeros should also be heavily prioritized because they're so short to encode
# so the estimate is correct for longer sequences, but the choice is not.
# like given the next n bytes are this, then we can achieve this amount of savings on the
# rest, but the next n bytes might not be those! Therefore the choice is sometimes wrong.

        if options and savings(best_option := max(options, key=savings)) > tolerance:
            # print("best: ", best_option)
            # print(savings(best_option)//8)
            block.insert(idx, stride_escape)
            if label_mode:
                for offset, num in enumerate(nums_to_bytes_ascii(best_option)):
                    block.insert(idx + 1 + offset, num)
                idx += 1 + len(nums_to_bytes_ascii(best_option))
            else:
                for offset, num in enumerate(nums_to_bytes(best_option)):
                    # print(byte_to_nums(nums_to_bytes(best_option)))
                    block.insert(idx + 1 + offset, num)
                idx += 1 + len(nums_to_bytes(best_option))
            # let's have sequence come after
            indices_to_remove = affected_indices(idx, *best_option)
            if best_option[0] == [0]*len(best_option[0]): # zero run
                for idy in range(idx, idx+len(best_option[0])):
                    indices_to_remove.append(idy)
            # print(indices_to_remove)
            block = [
                num
                for num_idx, num in enumerate(block)
                if num_idx not in indices_to_remove
            ]
            # print(best_option)
            idx += len(best_option[0])
        else:
            idx += 1
    return block


def expand(run, escape_nums, rest):
    _, stride, repeat, zero_seq = escape_nums
    new_piece = []
    idx = 0
    while repeat:
        new_piece.extend(rest[idx : idx + stride])
        idx += stride
        new_piece.extend(run)
        repeat -= 1
    # print(savings(escape_nums)//8)
    # <<<savings, seq_len, stride, count>>>
    # print('run', run, 'escape', escape_nums, 'expanded', new_piece)
    new_piece.extend(rest[idx:])
    return new_piece


def decompress_block(block, stride_escape):
    total_bits = sum(settings)
    # print(settings)
    total_bits += (8 - total_bits % 8) % 8
    bytes_needed = total_bits // 8

    chunks = split_list(block, stride_escape)
    while len(chunks) > 1:
        chunk = chunks[-1]
        numbers = byte_to_nums(chunk[:bytes_needed])
        seq_len = numbers[0]
        # print(numbers)
        if numbers[-1]: # zero sequence
            rest = chunk[bytes_needed:]
            run = [0]*seq_len
        else:
            rest = chunk[bytes_needed + seq_len :]
            run = chunk[bytes_needed : bytes_needed + seq_len]
        chunks.pop()
        new_chunk = chunks.pop() + run + expand(run, numbers, rest)
        chunks.append(new_chunk)
    return chunks[0]


if __name__ == "__main__":
    # block = sum([[0,0,0,num,num+1] for num in range(254)], start=[])
    block = sum([[0, 0] for num in range(20)], start=[])
    unused = (set(range(256)) - set(block)).pop()
    compressed_labeled = compress_block(block, unused, label_mode=True)
    compressed = compress_block(block, unused)
    # each of these spends 3 bytes (one for lit escape, 2 to encode the bits. Can try
    # subtracting one byte times all of these and also 2 bytes)

    csc_instances = compressed.count(compressed[0]) - 1
    # compressed size, compressed_size -1 byte each, compressed_size -2 bytes each
    print(
        len(compressed),
        len(compressed) - csc_instances,
        len(compressed) - 2 * csc_instances,
    )
    print(decompress_block(compressed, unused)==block)
    print(len(decompress_block(compressed, unused)))
