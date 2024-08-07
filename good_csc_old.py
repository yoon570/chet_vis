from itertools import takewhile
from copy import deepcopy
import numpy as np
from byteobj import Byte

METADATA_BITS = 16 # shared between pattern identification and repeat. either 8 or 16

class Pattern:
    def __init__(self, seq: tuple[int], huffman_code):
        self.huffman_code = huffman_code
        self.seq = seq
        self.length = len(seq)
        self.identity_length = len(list(filter(lambda x:x<=-2, self.seq)))
        self.match_part_length = max(self.seq.count(0), self.identity_length)
        self.repeat_bits = METADATA_BITS-len(huffman_code)

        self.total_savings = 0
        self.min_saving_count = 0
        while self.savings(self.min_saving_count) <= 0:
            self.min_saving_count += 1

    def __repr__(self):
        return f"{self.seq}"
    
    def pack_metadata(self, repeat):
        # should be made more robust and flexible later
        # that time is now
        # binary = str(patterns.index(self)) + bin(repeat)[2:].zfill(7)
        # print(repeat-self.min_saving_count)
        binary = self.huffman_code + bin(repeat-self.min_saving_count)[2:].zfill(METADATA_BITS-len(self.huffman_code))
        assert len(binary) == METADATA_BITS and 0 <= int(binary,2) < 2**METADATA_BITS
        return np.packbits([1 if val == '1' else 0 for val in binary]).tolist()
    
    def unpack_metadata(byte_vals):
        binary = np.unpackbits(np.frombuffer(bytes(byte_vals), dtype=np.uint8))
        binary = "".join('1' if val else '0' for val in binary).zfill(METADATA_BITS)
        # print(binary)
        matching_pattern = None
        for pattern in patterns:
            if binary.startswith(pattern.huffman_code):
                matching_pattern = pattern
        if matching_pattern is None:
            raise Exception
        repeat = int(binary[len(matching_pattern.huffman_code):], 2) + matching_pattern.min_saving_count
        
        return matching_pattern, repeat

    def match(self, page): # gives the number of matches (repeat len including first match)
        # requires that at least one of the -1 values are not zero to avoid overlapping matches?
        # can tweak this as required potentially.
        match_dict = {}
        for pattern_num, page_num in zip(self.seq, page):
            if self.identity_length and pattern_num <= -2:
                match_dict[pattern_num] = page_num

        count = 0
        idx = 0
        # if patterns.index(self) == 4:
            # print(match_dict)
        other_nums = []
        # want to add a filter stage such that if -1 values are zero with too high a frequency, the pattern is ignored
        # while idx < len(page) and count < 127: # ignore 12 for now
        # i guess you'd never store a count of zero? Should it start where it's profitable?
        # can adjust that now
        while idx < len(page) and (count+1-self.min_saving_count).bit_length() <= self.repeat_bits: 
            next_chunk = page[idx:idx+self.length]
            if len(next_chunk) != self.length:
                break
            if self.identity_length: # not zeros
                if any(match_dict[pattern_num] != page_num for pattern_num, page_num in zip(self.seq, next_chunk[:self.identity_length])):
                    break
            else: # zeros
                if any(pattern_num != page_num for pattern_num, page_num in zip(self.seq, next_chunk[:self.match_part_length])):
                    break
            # potentially remove later
            other_nums.extend(next_chunk[self.match_part_length:])
            count += 1
            idx += self.length

        # THRESHOLD = .3
        # THRESHOLD = 1 # turn this off for a sec

        # if other_nums and other_nums.count(0)/len(other_nums) > THRESHOLD:
            # return 0
        
        return count
    def uncompressed_size(self, freq):
        return self.match_part_length * freq # bytes
    def compressed_size(self): # can be used for splitting. first byte after the escape can encode just the idx, then you can grab this value
        # size = 3 # escape byte, pattern byte, then repeat length
        size = 1 + METADATA_BITS//8 # escape byte, pattern + repeat length byte
        if self.identity_length: # requires identity bytes
            size += self.identity_length
        return size
    def savings(self, freq):
        return self.uncompressed_size(freq) - self.compressed_size()
    
    def compress(self, page):
        occurrences = self.match(page)
        new_page = []
        encoded_bytes = self.pack_metadata(occurrences)
        # new_page.append(index) # put the index first
        # new_page.append(occurrences) # next repeats
        new_page.extend(encoded_bytes)
        new_page.extend(page[:self.identity_length]) # next identity (if necessary)
        # match will start from index 0 of the page
        # add the rest of the page minus the missing parts
        # the identity is already added, so we can just remove the patterned parts
        idx = 0
        for _ in range(occurrences):
            new_page.extend(page[idx+self.match_part_length:idx+self.length])
            idx += self.length

        self.total_savings += self.savings(occurrences)

        return new_page + page[idx:]
    # forgot there's weird issues with splitting the page up, so will probably have to remake a version of that 
    # function. Needs to consider variable width though which will be a little interesting
    def decompress(self, page, chunk_bs): # this will be done back to front assuming that the first byte is the literal escape
        # the first two bytes are kind of useless at this stage (just escape and index)
        metadata_length = METADATA_BITS//8
        pattern, occurrences = Pattern.unpack_metadata(page[1:1+metadata_length]) 
        # occurrences = page[2]
        if self.identity_length:
            identity = page[1+metadata_length:1+metadata_length+self.match_part_length]
            identity_bs = chunk_bs[1+metadata_length:1+metadata_length+self.match_part_length]
                
        else:
            identity = [0]*self.match_part_length
            # Generic -1 flag to show that there is no identity portion ot metadata
            identity_bs = [Byte(0, -1, [], 0)]*self.match_part_length

        page = page[1+metadata_length+self.identity_length:]
        chunk_bs = chunk_bs[1+metadata_length+self.identity_length:]
        
        for which_p, patt in enumerate(patterns):
            if patt == pattern:
                    # Adding 1 to the marker so it's never indexed @ 0
                stage_marker = which_p + 1
                break
                
        if stage_marker == 1:
            stage_marker = -1
            
        for part in identity_bs:
            part.stage1 = stage_marker

        # should fill in the expansion an occurrences amount of times starting with occurrence then the stride
        new_page = []
        new_chunk_bs = []

        idx = 0
        for _ in range(occurrences):
            new_page.extend(identity)
            new_page.extend(page[idx:idx+self.length-self.match_part_length])
            
            new_chunk_bs.extend(identity_bs)
            new_chunk_bs.extend(chunk_bs[idx:idx+self.length-self.match_part_length])
            idx += self.length-self.match_part_length

        new_page.extend(page[idx:])
        new_chunk_bs.extend(chunk_bs[idx:])
        return new_page, new_chunk_bs

def split_instances(page, escape, post_csc_bs):
    # turn it into a list split by constant stride
    metadata_length = METADATA_BITS//8
    chunks = [[]]
    chunks_bs = [[]]
    idx = 0
    while idx < len(page):
        if page[idx] == escape: 
            # now split off a new chunk. technically it can be very first though, so just remove the initial
            # empty list if that's the case
            chunks.append([])
            chunks_bs.append([])
            pattern, _ = Pattern.unpack_metadata(page[idx+1:idx+1+metadata_length])
            size = pattern.compressed_size() # includes stride
            chunks[-1].extend(page[idx:idx+size]) # now it's safe
            for i in range(size):
                chunks_bs[-1].append(post_csc_bs[idx + i])
            idx += size
        else:
            chunks[-1].append(page[idx])
            chunks_bs[-1].append(post_csc_bs[idx])
            idx += 1
    if not chunks[0]:
        chunks = chunks[1:]
        chunks_bs = chunks_bs[1:]
    return chunks, chunks_bs

# based on the data, it's pretty natural to have the first bit represent run of zeros
# or not, if not, then the second bit picks between idx 2 and 12. Can also give up on 
# 12 which I think is best because of how easy 0 and 2 are in hardware.

lazy_pattern_huffman_codes = iter(bin(num)[2:].zfill(5) for num in range(32))
patterns = [
    Pattern((0,), next(lazy_pattern_huffman_codes)), # 0 RLE
    Pattern((-2,), next(lazy_pattern_huffman_codes)), # num RLE

    Pattern((0,-1), next(lazy_pattern_huffman_codes)), # alternating num 
    Pattern((-2,-1), next(lazy_pattern_huffman_codes)), # alternating with some number # can maybe add just alternating, but I think bse handles

    Pattern((0,-1,-1,-1), next(lazy_pattern_huffman_codes)), # 0 every 4
    Pattern((-2,-1,-1,-1), next(lazy_pattern_huffman_codes)), # num every 4 
    Pattern((0,0,-1,-1), next(lazy_pattern_huffman_codes)), # two 0s every 4
    Pattern((-2,-3,-1,-1), next(lazy_pattern_huffman_codes)), # two nums every 4

    Pattern((0,)+(-1,)*7, next(lazy_pattern_huffman_codes)), # 0 every 8 
    Pattern((-2,)+(-1,)*7, next(lazy_pattern_huffman_codes)), # num every 8
    Pattern((0,0)+(-1,)*6, next(lazy_pattern_huffman_codes)), # two 0s every 8 
    Pattern((-2,-3)+(-1,)*6, next(lazy_pattern_huffman_codes)), # two nums every 8
    Pattern((0,)*4+(-1,)*4, next(lazy_pattern_huffman_codes)), # four 0s every 8
    Pattern((-2,-3,-4,-5)+(-1,)*4, next(lazy_pattern_huffman_codes)), # four nums every 8

    Pattern((0,)+(-1,)*15, next(lazy_pattern_huffman_codes)), # 0 every 16
    Pattern((-2,)+(-1,)*15, next(lazy_pattern_huffman_codes)), # num every 16
    Pattern((0,0)+(-1,)*14, next(lazy_pattern_huffman_codes)), # two 0s every 16
    Pattern((-2,-3)+(-1,)*14, next(lazy_pattern_huffman_codes)), # two nums every 16
    Pattern((0,)*4+(-1,)*12, next(lazy_pattern_huffman_codes)), # four 0s every 16
    Pattern((-2,-3,-4,-5)+(-1,)*12, next(lazy_pattern_huffman_codes)), # four nums every 16
    Pattern((0,)*8+(-1,)*8, next(lazy_pattern_huffman_codes)), # eight 0s every 16
    Pattern((-2,-3,-4,-5,-6,-7,-8,-9)+(-1,)*8, next(lazy_pattern_huffman_codes)), # eight nums every 16
]

def report_pattern_savings(): # reports and resets
    pattern_savings = []
    for pattern in patterns:
        pattern_savings.append(pattern.total_savings)
        pattern.total_savings = 0
    return pattern_savings

def read_file_in_chunks(filename, chunk_size=4096):
    with open(filename, "rb") as file:
        chunk_iter = iter(lambda: file.read(chunk_size), b"")
        return list(takewhile(lambda chunk: chunk, chunk_iter))

# modify it to look in the local forward the next, say 8?
LOCAL_WINDOW = 16 # size so 1 is the lowest it should go
def compress_block(block, stride_escape, tolerance=0, label_mode=False):
    block = deepcopy(block)
    idx = 0
    while idx < len(block):
        options = []
        for skip in range(LOCAL_WINDOW):
            curr_idx = idx + skip
            for pattern in patterns:
                # savings = pattern.savings(pattern.match(block[curr_idx:]))
                savings = pattern.savings(pattern.match(block[curr_idx:]))
                # print(savings, pattern)
                if savings > tolerance:
                    options.append(((pattern, savings), skip))
        if options:
            (best_pattern, _), skip = max(options, key=lambda x:x[0][1]) # best by savings
            #print(best_pattern, _)
            idx += skip
            new_block_end = best_pattern.compress(block[idx:]) 
            block[idx:] = [stride_escape] + new_block_end
            idx += best_pattern.compressed_size()
        else:
            idx += 1
    return block

def decompress_block(block, stride_escape, post_csc_bs):
    chunks_pair = split_instances(block, stride_escape, post_csc_bs)
    chunks = chunks_pair[0]
    chunks_bs = chunks_pair[1]
    metadata_length = METADATA_BITS//8
    
    if chunks[0] and chunks[0][0] == stride_escape:
        chunks.insert(0, [])
        chunks_bs.insert(0, [])
    while len(chunks) > 1:
        chunk = chunks[-1]
        chunk_bs = chunks_bs[-1]
        pattern, _ = Pattern.unpack_metadata(chunk[1:1+metadata_length])
        decompressed_chunk_pair = pattern.decompress(chunk, chunk_bs)
        decompressed_chunk = decompressed_chunk_pair[0]
        decompressed_chunk_bs = decompressed_chunk_pair[1]
        chunks.pop()
        chunks_bs.pop()
        new_chunk = chunks.pop() + decompressed_chunk
        new_chunk_bs = chunks_bs.pop() + decompressed_chunk_bs
        chunks.append(new_chunk)
        chunks_bs.append(new_chunk_bs)
    return chunks[0], chunks_bs[0]

if __name__ == "__main__":
    # this example doesn't work?!

    block = sum([[0,num+2,0,num+1] for num in range(3)], start=[]) + [0]*100
    # block = sum([[num,0,0,0] for num in range(254)], start=[])
    # block = sum([[1,0] for num in range(254)], start=[])
    # block = sum([[1,2] for num in range(254)], start=[])
    # block = sum([[3,(num+1)//2,(num+2)//2,(num+3)//2] for num in range(254)], start=[])
    # block = sum([[0,0] for num in range(20)], start=[])
    unused = (set(range(256)) - set(block)).pop()
    # compressed_labeled = compress_block(block, unused, label_mode=True)
    compressed = compress_block(block, unused)
    print(compressed)
    print(len(compressed))
    # print(decompress_block(compressed, unused))
    # print(block)
    print(decompress_block(compressed, unused) == block)
    # each of these spends 3 bytes (one for lit escape, 2 to encode the bits. Can try
    # subtracting one byte times all of these and also 2 bytes)# 

    # compressed size, compressed_size -1 byte each, compressed_size -2 bytes each