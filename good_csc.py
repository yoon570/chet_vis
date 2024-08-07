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
        span = 0
        while idx < len(page) and (count+1-self.min_saving_count).bit_length() <= self.repeat_bits: 
            next_chunk = page[idx:idx+self.length]
            # should also make it apply when there aren't enough other bytes to fill.
            # if len(next_chunk) != self.length:
            if len(next_chunk) < self.match_part_length:
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
            span += self.length
            idx += self.length
        if span:
            span -= (self.length - self.match_part_length)

        # THRESHOLD = .3
        # THRESHOLD = 1 # turn this off for a sec

        # if other_nums and other_nums.count(0)/len(other_nums) > THRESHOLD:
            # return 0
        
        return count, span
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
        occurrences, span = self.match(page)
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
            chunks_bs[-1].extend(post_csc_bs[idx:idx+size])
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
# LOCAL_WINDOW = 1 # size so 1 is the lowest it should go
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
                occurrences, span = pattern.match(block[curr_idx:])
                savings = pattern.savings(occurrences)

                if span:
                    # heuristic = savings
                    # heuristic = savings/span
                    heuristic = savings/span**.5
                else:
                    heuristic = 0

                if savings > tolerance:
                    options.append(((pattern, heuristic), skip))
        if options:
            (best_pattern, _), skip = max(options, key=lambda x:x[0][1]) # best by savings
            # print(best_pattern, _)
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
        chunks_bs.insert(0,[])
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
    block = """18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 44 27 b2 18 56 00 00 c6 6d 34 80 
b7 40 da 3f 9a 99 99 99 99 99 e9 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 ba 49 0c 02 2b 47 56 40 73 68 91 ed 7c f3 64 40 4a 0c 02 2b 87 06 61 40 00 00 00 00 00 00 00 00 50 50 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 
02 00 00 00 02 00 00 00 0d 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 44 27 b2 18 56 00 00 c4 3f 08 e4 14 1d e3 3f 33 33 33 33 33 33 fb 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ac 1c 5a 64 
3b b7 56 40 e7 fb a9 f1 d2 95 64 40 a0 1a 2f dd 24 d6 60 40 00 00 00 00 00 00 00 00 70 50 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 0c 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 44 27 b2 18 56 00 00 ef 04 cf 9f 3c 2c e2 bf 00 00 00 00 00 00 f8 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4a 0c 02 2b 87 96 56 40 fc a9 f1 d2 4d 76 64 40 c9 76 be 9f 
1a e7 60 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 41 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00 08 00 00 00 
09 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 21 00 00 00 00 00 00 00 54 48 52 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 20 32 35 35 
39 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 4e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 41 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 42 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 42 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 47 32 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 47 32 31 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 47 32 32 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 47 32 33 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 4f 47 31 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 47 31 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 4f 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 91 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 5b 27 b2 18 56 00 00 30 5b 27 b2 
18 56 00 00 81 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 e0 76 24 b2 18 56 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 0e 00 00 00 00 00 00 00 d0 5a 27 b2 18 56 00 00 20 51 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 b1 09 00 00 00 00 00 00 50 5b 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 
02 00 00 00 01 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 ed 9e 3c 2c d4 9a da bf cd cc cc cc cc cc f8 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 e3 a5 9b c4 
20 b0 56 40 c3 f5 28 5c 8f 9e 64 40 f8 53 e3 a5 9b ac 60 40 00 00 00 00 00 00 00 00 70 5b 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 d8 12 f2 41 cf 66 d1 3f cd cc cc cc cc cc f4 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 14 ae 47 e1 7a cc 56 40 77 be 9f 1a 2f b9 64 40 a2 45 b6 f3 
fd a0 60 40 00 00 00 00 00 00 00 00 90 5b 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 04 00 00 00 00 00 00 00 03 00 00 00 04 00 00 00 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 90 50 27 b2 18 56 00 00 0a f9 a0 67 b3 ea a3 bf 33 33 33 33 33 33 fb 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 64 3b df 4f 8d 7f 56 40 ec 51 b8 1e 85 83 64 40 f4 fd d4 78 e9 8e 60 40 00 00 00 00 00 00 00 00 b0 5b 27 b2 
18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 0f 0b b5 a6 
79 c7 b9 3f cd cc cc cc cc cc f4 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 6a bc 74 93 18 3c 56 40 ae 47 e1 7a 14 8a 64 40 be 9f 1a 2f dd 94 60 40 00 00 00 00 00 00 00 00 d0 5b 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 
04 00 00 00 02 00 00 00 05 00 00 00 06 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 6d 56 7d ae b6 62 d7 3f 33 33 33 33 33 33 fb 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 79 e9 26 31 
08 94 56 40 f4 fd d4 78 e9 8e 64 40 00 00 00 00 00 60 60 40 00 00 00 00 00 00 00 00 f0 5b 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 04 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 22 fd f6 75 e0 9c 71 3f cd cc cc cc cc cc f4 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6a bc 74 93 18 6c 56 40 b2 9d ef a7 c6 7b 64 40 98 6e 12 83 
c0 4a 60 40 00 00 00 00 00 00 00 00 10 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 04 00 00 00 04 00 00 00 07 00 00 00 08 00 00 00 09 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 90 50 27 b2 18 56 00 00 f7 e4 61 a1 d6 34 cf bf 33 33 33 33 33 33 fb 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 19 04 56 0e 2d 82 56 40 83 c0 ca a1 45 be 64 40 9c c4 20 b0 72 58 60 40 00 00 00 00 00 00 00 00 30 5c 27 b2 
18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 1e 38 67 44 
69 6f b0 3f cd cc cc cc cc cc f4 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 27 31 08 ac 1c aa 56 40 3d 0a d7 a3 70 d1 64 40 8b 6c e7 fb a9 6d 60 40 00 00 00 00 00 00 00 00 50 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 
01 00 00 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 1e 38 67 44 69 6f b0 3f cd cc cc cc cc cc f4 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 71 3d 0a d7 
a3 90 56 40 ee 7c 3f 35 5e c6 64 40 9e ef a7 c6 4b 37 60 40 00 00 00 00 00 00 00 00 70 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 06 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 1e 38 67 44 69 6f b0 3f cd cc cc cc cc cc f4 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f4 fd d4 78 e9 3e 56 40 a2 45 b6 f3 fd c4 64 40 6d e7 fb a9 
f1 5e 60 40 00 00 00 00 00 00 00 00 90 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 04 00 00 00 0b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 90 50 27 b2 18 56 00 00 12 26 b8 76 9c a2 e5 bf 00 00 00 00 00 00 f8 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4e 62 10 58 39 ec 56 40 a0 1a 2f dd 24 86 64 40 fa 7e 6a bc 74 57 60 40 00 00 00 00 00 00 00 00 b0 5c 27 b2 
18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 c6 6d 34 80 
b7 40 da 3f 9a 99 99 99 99 99 e9 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 29 5c 8f c2 f5 f8 56 40 68 91 ed 7c 3f 8d 64 40 0a d7 a3 70 3d 3a 60 40 00 00 00 00 00 00 00 00 d0 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 
02 00 00 00 02 00 00 00 0d 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 c4 3f 08 e4 14 1d e3 3f 33 33 33 33 33 33 fb 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2b 87 16 d9 
ce 8f 56 40 64 3b df 4f 8d 53 64 40 b2 9d ef a7 c6 93 60 40 00 00 00 00 00 00 00 00 f0 5c 27 b2 18 56 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 0c 00 00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 50 27 b2 18 56 00 00 ef 04 cf 9f 3c 2c e2 bf 00 00 00 00 00 00 f8 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 3f 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 fc a9 f1 d2 4d c2 56 40 21 b0 72 68 91 45 64 40 4a 0c 02 2b 
87 ae 60 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 41 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00 08 00 00 00 
09 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 21 00 00 00 00 00 00 00 54 48 52 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 20 32 35 36 
30 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 4e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 00 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 41 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 42 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 42 00 00 
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 43 47 32 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00 48 47 32 31"""
    block = block.replace('\n', '')
    block = [int(num, 16) for num in block.split()]
    print(len(block))

    unused = (set(range(256)) - set(block)).pop()
    compressed = compress_block(block, unused)
    print(compressed)
    print(len(compressed))
    # print(decompress_block(compressed, unused))
    # print(block)
    print(decompress_block(compressed, unused) == block)
    print(report_pattern_savings())
    # each of these spends 3 bytes (one for lit escape, 2 to encode the bits. Can try
    # subtracting one byte times all of these and also 2 bytes)# 

    # compressed size, compressed_size -1 byte each, compressed_size -2 bytes each