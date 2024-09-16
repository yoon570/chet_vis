from itertools import batched, islice, takewhile
from collections import defaultdict, deque, Counter
import glob
from test_alg import parallel_test_compression_ratio_chunks, serial_test_compression_ratio_chunks
from math import log2
import pickle

def read_file_in_chunks(filename, chunk_size=4096):
    with open(filename, "rb") as file:
        chunk_iter = iter(lambda: file.read(chunk_size), b"")
        return list(takewhile(lambda chunk: chunk, chunk_iter))

from typing import List

from abc import ABC, abstractmethod
from dataclasses import dataclass

def bse_savings_get():
    return bse_applications

def csc_savings_get():
    return csc_applications

bse_applications = [0] * 63
csc_applications = [0] * 4
# Add into here the stages and what sequences affected it
class MetaByte(ABC):
    def __init__(self):
        self.stage1 = []
        self.stage2 = []

    @classmethod
    @abstractmethod
    def get_prefix(cls) -> str:
        pass
    @classmethod
    @abstractmethod
    def get_max_val(cls) -> int:
        pass
    @classmethod
    @abstractmethod
    def get_min_val(cls) -> int:
        pass
    @classmethod
    @abstractmethod
    def decode(cls, value):
        pass
    
    def set_metadata(self, csc_metadata, bse_metadata):
        self.stage1 = csc_metadata
        self.stage2 = bse_metadata
    def get_metadata(self):
        return (self.stage1, self.stage2)
        
    @property
    @abstractmethod
    def repr_str(self) -> str:
        pass
    @property
    @abstractmethod
    def num_encoding(self) -> int:
        pass
    def __repr__(self) -> str:
        return self.repr_str
    def __hash__(self) -> int:
        return self.num_encoding
    def __eq__(self, value: object) -> bool:
        return isinstance(value, MetaByte) and self.num_encoding == value.num_encoding
    def __gt__(self, value: object) -> bool:
        return isinstance(value, MetaByte) and self.num_encoding > value.num_encoding

class LitByte(MetaByte):
    def __init__(self, value):
        super().__init__()
        assert 0 <= value < 2**8
        self._value = value
    @classmethod
    def get_prefix(cls) -> str:
        return '0'
    @classmethod
    def get_max_val(cls) -> int:
        return 255
    @classmethod
    def get_min_val(cls) -> int:
        return 0
    @classmethod
    def decode(cls, value:str):
        assert len(value) == 9
        return cls(int(value[len(cls.get_prefix()):], 2))
    @property
    def repr_str(self) -> str:
        return f"[Literal: {self._value:02x}]"
    @property
    def num_encoding(self) -> int:
        binary_value = f"{self._value:08b}"
        nine_bit_binary = self.get_prefix() + binary_value
        assert len(nine_bit_binary) == 9
        return int(nine_bit_binary, 2)

class BSESub(MetaByte):
    def __init__(self, sub_idx):
        super().__init__()
        assert 0 <= sub_idx < 64
        self._sub_idx = sub_idx
    @classmethod
    def get_prefix(cls) -> str:
        return '111'
    @classmethod
    def get_max_val(cls) -> int:
        return 63
    @classmethod
    def get_min_val(cls) -> int:
        return 0
    @classmethod
    def decode(cls, value:str):
        assert len(value) == 9
        return cls(int(value[len(cls.get_prefix()):], 2))
    @property
    def repr_str(self) -> str:
        return f"[BSE Sub: {self._sub_idx}]"
    @property
    def num_encoding(self) -> int:
        binary_value = f"{self._sub_idx:06b}"
        nine_bit_binary = self.get_prefix() + binary_value
        assert len(nine_bit_binary) == 9
        return int(nine_bit_binary, 2)
    @property
    def sub_index(self) -> int:
        return self._sub_idx

class CSC0Byte(MetaByte):
    def __init__(self, repeats):
        super().__init__()
        assert self.get_min_val() <= repeats <= self.get_max_val()
        self._repeats = repeats
    @classmethod
    def get_max_val(cls) -> int:
        repeat_bits = 9 - len(cls.get_prefix())
        return cls.get_min_val() + 2**repeat_bits - 1
    @classmethod
    def get_min_val(cls) -> int:
        return 2
    @classmethod
    def decode(cls, value:str):
        assert len(value) == 9
        return cls(int(value[len(cls.get_prefix()):], 2) + cls.get_min_val())
    @classmethod
    def compress(cls, chunk:List[MetaByte]) -> List[MetaByte]:
        count, _ = cls.match(chunk)
        new_page = []
        encoded_byte = cls(count)
        new_page.append(encoded_byte)
        idx = 0
        for _ in range(count):
            new_page.extend(chunk[idx+len(cls.get_match_part()):idx+cls.get_stride()])
            idx += cls.get_stride()
        new_page += chunk[idx:]
        return new_page
    @classmethod
    def decompress(cls, chunk: List[MetaByte], part_idx) -> List[MetaByte]:
        csc_byte: CSC0Byte = chunk[0]
        chunk = chunk[1:]

        new_page = []
        idx = 0
        for _ in range(csc_byte.repeats):
            # Try this? This should append metadata to each csc byte that gets attached here
            # At least that's the hope
            matchpart = deepcopy(csc_byte.get_match_part())
            for part in matchpart:
                part_metadata = part.get_metadata()
                csc_metadata = part_metadata[0]
                csc_metadata.append(part_idx)
                part.set_metadata(csc_metadata, part_metadata[1])
            new_page.extend(matchpart)
            new_page.extend(chunk[idx:idx+csc_byte.get_stride()-len(csc_byte.get_match_part())])
            idx += csc_byte.get_stride()-len(csc_byte.get_match_part())

        new_page.extend(chunk[idx:])
        return new_page

    @property
    def repeats(self) -> int:
        return self._repeats

    @property
    def num_encoding(self) -> int:
        repeat_bits = 9-len(self.get_prefix())
        binary_repeats = bin(self._repeats-self.get_min_val())[2:].zfill(repeat_bits)
        nine_bit_binary = self.get_prefix() + binary_repeats
        assert len(nine_bit_binary) == 9
        return int(nine_bit_binary, 2)
    @classmethod
    @abstractmethod
    def get_match_part(self) -> List[LitByte]:
        pass
    @classmethod
    @abstractmethod
    def get_stride(self) -> int:
        pass
    @classmethod
    def match(cls, page: List[MetaByte]): # gives the number of matches starting from current location
        count = 0
        idx = 0
        span = 0
        while idx < len(page) and count < cls.get_max_val(): 
            next_candidate = page[idx:idx+cls.get_stride()]
            if len(next_candidate) < len(cls.get_match_part()):
                break
            if any(pattern_num != page_num for pattern_num, page_num in zip(cls.get_match_part(), next_candidate[:len(cls.get_match_part())])):
                break
            count += 1
            span += cls.get_stride()
            idx += cls.get_stride()
        if span:
            span -= cls.get_stride()-len(cls.get_match_part())
        return count, span
    @classmethod
    def savings(cls, matches: int): # in bits
        # based on the fact that everything becomes 9 bits
        if matches < cls.get_min_val():
            return 0
        uncompressed_size = matches * len(cls.get_match_part()) * 9
        compressed_size = 9
        return uncompressed_size - compressed_size

class RLE0Byte(CSC0Byte):
    @classmethod
    def get_prefix(cls) -> str:
        return '100'
    @property
    def repr_str(self) -> str:
        return f"[RLE0 repeats: {self._repeats}]"
    @classmethod
    def get_match_part(cls) -> List[LitByte]:
        return [LitByte(0)]
    @classmethod
    def get_stride(cls) -> int:
        return 1
    @classmethod
    def get_min_val(cls) -> int:
        return 3 # to not conflict with others even when going first
    # might leave some zero pairs for bse, but oh well.
    
class ALT0Byte(CSC0Byte):
    @classmethod
    def get_prefix(cls) -> str:
        return '101'
    @property
    def repr_str(self) -> str:
        return f"[ALT0 repeats: {self._repeats}]"
    @classmethod
    def get_match_part(cls) -> List[LitByte]:
        return [LitByte(0)]
    @classmethod
    def get_stride(cls) -> int:
        return 2

class DB04Byte(CSC0Byte):
    @classmethod
    def get_prefix(cls) -> str:
        return '1100'
    @property
    def repr_str(self) -> str:
        return f"[DB04 repeats: {self._repeats}]"
    @classmethod
    def get_match_part(cls) -> List[LitByte]:
        return [LitByte(0), LitByte(0)]
    @classmethod
    def get_stride(cls) -> int:
        return 4
       
class DB08Byte(CSC0Byte):
    @classmethod
    def get_prefix(cls) -> str:
        return '1101'
    @property
    def repr_str(self) -> str:
        return f"[DB08 repeats: {self._repeats}]"
    @classmethod
    def get_match_part(cls) -> List[LitByte]:
        return [LitByte(0), LitByte(0)]
    @classmethod
    def get_stride(cls) -> int:
        return 8
    @classmethod
    def get_min_val(cls) -> int:
        return 1 # try to get it to take care of double zeros. just to try
    
def decode_value(value):
    assert 0 <= value < 512
    binary = f"{value:09b}"
    for metabyte_type in [LitByte, BSESub, RLE0Byte, ALT0Byte, DB04Byte, DB08Byte]:
        if binary.startswith(metabyte_type.get_prefix()):
            return metabyte_type.decode(binary)
    assert False, "unreachable"

# used before first encoder stage
def pad_input(page: bytes) -> List[MetaByte]: # pre-rle stage, convert to metabyte arr.
    metabyte_page = []
    for byte_val in page:
        metabyte_page.append(LitByte(byte_val))
    return metabyte_page

# used after last decoder stage
def unpad_output(metabyte_page: List[MetaByte]):
    regular_page = [value.num_encoding for value in metabyte_page]
    # makes sure they were all literals. relies on zero first bit for lits
    assert all(0 <= val < 256 for val in regular_page)
    return bytes(regular_page)

def chunk_input(metabyte_page: List[MetaByte], chunksize:int) -> List[List[MetaByte]]:
    return list(map(list, batched(metabyte_page, chunksize)))

def unchunk_output(metabyte_chunks: List[List[MetaByte]]) -> List[MetaByte]:
    return sum(metabyte_chunks, start=[])

# operates independently on a single chunk
from copy import deepcopy

def csc0_encode(metabyte_chunk: List[MetaByte]) -> List[MetaByte]:
    metabyte_chunk = deepcopy(metabyte_chunk) # so things stay pure
    # patterns: List[CSC0Byte] = [RLE0Byte, ALT0Byte, DB04Byte, DB08Byte]
    # this order?
    # rle needs to be first though without applying for each byte
    # patterns: List[CSC0Byte] = [DB04Byte, DB08Byte, RLE0Byte, ALT0Byte]
    patterns: List[CSC0Byte] = [RLE0Byte, DB04Byte, DB08Byte, ALT0Byte]
    for pattern in patterns:
        idx = 0
        while idx < len(metabyte_chunk):
            count, _ = pattern.match(metabyte_chunk[idx:])
            savings = pattern.savings(count)
            if savings > 0: # actually saves
                metabyte_chunk[idx:] = pattern.compress(metabyte_chunk[idx:])
            idx += 1 # advance by one byte no matter what
    return metabyte_chunk

def csc0_decode(metabyte_chunk: List[MetaByte]) -> List[MetaByte]:
    metabyte_chunk = deepcopy(metabyte_chunk) 
    patterns: List[CSC0Byte] = [RLE0Byte, DB04Byte, DB08Byte, ALT0Byte][::-1]
    for patt_idx, pattern in enumerate(patterns):
        idx = len(metabyte_chunk)-1
        while idx >= 0: # go in reverse order
            metabyte = metabyte_chunk[idx]
            if isinstance(metabyte, pattern):
                # RLE0 = 0, DB04 = 1, etc.
                csc_applications[3 - patt_idx] += 1
                metabyte_chunk[idx:] = CSC0Byte.decompress(metabyte_chunk[idx:], 3 - patt_idx)
            idx -= 1
    return metabyte_chunk

def apply_chunkwise(func, metabyte_chunks: List[List[MetaByte]], *args):
    return [func(metabyte_page, *args) for metabyte_page in metabyte_chunks]

# can worry about padding to nearest byte later
def metabyte_page_to_binary(metabyte_page: List[MetaByte]):
    binary_arr = []
    for metabyte in metabyte_page:
        binary_arr.append(f"{metabyte.num_encoding:09b}")
    return "".join(binary_arr)

def binary_to_metabyte_page(binary):
    metabyte_page = []
    for batch in batched(binary, 9):
        metabyte_page.append(decode_value(int("".join(batch), 2)))
    return metabyte_page

def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)

class SubMetaByteGenerator:
    def __init__(self):
        self.current = None
        self.next_idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.next_idx < 63:
            self.next_idx += 1
            return BSESub(self.next_idx-1)
        raise StopIteration

def total_length(metabyte_chunks):
    return sum(len(chunk) for chunk in metabyte_chunks)

def bse_iteration(metabyte_chunks, metabyte_sub, remaining_bits):
    util_coefs = {2: 9/20, 3: 18/29, 4: 27/38, 5: 36/47}
    dict_sizes = {2: 20, 3: 29, 4: 38, 5: 47}
    c = defaultdict(lambda: -1)
    for chunk in metabyte_chunks:
        for seq_len in range(2, 6):
            if dict_sizes[seq_len] <= remaining_bits:
                for seq in sliding_window(chunk, seq_len):
                    c[seq] += util_coefs[len(seq)]
    if c:
        best = max(c, key=lambda x:c[x])
    else:
        return (), metabyte_chunks
    new_metabyte_chunks = [] 
    for chunk in metabyte_chunks:
        new_chunk = []
        idx = 0
        while idx < len(chunk):
            if best == tuple(chunk[idx:idx+len(best)]):
                new_chunk.append(metabyte_sub)
                idx += len(best)
            else:
                new_chunk.append(chunk[idx])
                idx += 1
        new_metabyte_chunks.append(new_chunk)

    if total_length(metabyte_chunks)*9 < total_length(new_metabyte_chunks)*9 + 2 + len(best)*9:
        return (), metabyte_chunks
    return best, new_metabyte_chunks

def bse_encode_9b(metabyte_chunks: List[List[MetaByte]], memory_blocks: int):
    subs = []
    compressed = metabyte_chunks
    total_bits = memory_blocks * 64 * 8 - 6 #(-6 for count bits)

    for sub_byte in SubMetaByteGenerator():
        old = compressed
        sub, compressed = bse_iteration(compressed, sub_byte, total_bits)
        if not sub or total_bits-(len(sub)*9 + 2) < 0:
            compressed = old
            break
        total_bits -= len(sub) * 9 + 2
        subs.append(sub)

    count = len(subs)
    assert 0<=count<64
    count_bits = f"{count:06b}"
    sub_length_bits = [bin(length-2)[2:].zfill(2) for length in map(len, subs)]
    sub_length_bits = "".join(sub_length_bits)

    sub_bits = [f"{metabyte.num_encoding:09b}" for sub in subs for metabyte in sub]
    sub_bits = "".join(sub_bits)
    dict_bits = count_bits + sub_length_bits + sub_bits 
    assert len(dict_bits) <= memory_blocks*64*8
    return compressed, dict_bits

# Dictionary parsing code
def extract_sequences(dict_bits: str):
    idx = 0
    count = int(dict_bits[:6], 2)
    idx += 6
    lengths = []
    for _ in range(count):
        lengths.append(int(dict_bits[idx:idx+2], 2) + 2)
        idx += 2
    sequences = []
    for length in lengths:
        sequence = []
        for _ in range(length):
            sequence.append(decode_value(int(dict_bits[idx:idx+9], 2)))
            idx += 9
        sequences.append(sequence)
        
    expansions = [[LitByte(value=0) for _ in range(5)] for _ in range(64)]
    seq_lengths = [0] * 64

    for sub_byte, seq in zip(SubMetaByteGenerator(), sequences):
        seq_lengths[sub_byte.sub_index] = len(seq)
        for idx, metabyte in enumerate(seq[::-1]): # reverse early
            expansions[sub_byte.sub_index][idx] = metabyte
        
    return sequences, expansions, seq_lengths

from copy import deepcopy

def bse_decode_9b(metabyte_page: List[MetaByte], dict_bits: str):
    
    # Moved dictionary parsing code so I could access it from the CHET runner
    _, expansions, seq_lengths = extract_sequences(dict_bits)

    page = deque(metabyte_page)
    output = []

    # This is the same as the 8bit, touch here for BSE
    while page:
        num = page.popleft()
        if isinstance(num, BSESub) and seq_lengths[num.sub_index]: # on expansion list
            # Extract metadata
            bse_applications[num.sub_index] += 1
            num_metadata = num.get_metadata()
            new_stage2 = deepcopy(num_metadata[1])
            new_stage2.append(num.sub_index)
            for idx in range(seq_lengths[num.sub_index]):

                # Create new metabyte
                newbyte = deepcopy(expansions[num.sub_index][idx])
                # Keep stage 1 metadata same, append new replacement candidate

                newbyte.set_metadata(deepcopy(num_metadata[0]), new_stage2)
                # Append newbyte to page
                page.appendleft(newbyte)
        else:
            output.append(num)
    return output
# 1 blk
static_tree = {22: '00000000', 187: '00000001', 76: '00000010', 152: '00000011', 99: '0000010', 110: '0000011', 453: '000010', 63: '0000110', 257: '0000111', 82: '00010000', 258: '00010001', 88: '00010010', 193: '00010011', 320: '0001010000', 271: '0001010001', 327: '0001010010000', 341: '0001010010001000', 390: '0001010010001001', 366: '000101001000101', 367: '0001010010001100', 306: '0001010010001101', 339: '000101001000111', 420: '000101001001', 384: '00010100101', 469: '0001010011', 85: '00010101', 12: '00010110', 11: '00010111', 24: '00011000', 34: '00011001', 70: '00011010', 33: '00011011', 108: '0001110', 255: '00011110', 9: '00011111', 114: '0010000', 119: '00100010', 248: '00100011', 72: '00100100', 95: '00100101', 0: '0010011', 450: '001010', 452: '001011', 317: '00110000000000', 361: '00110000000001000', 409: '00110000000001001', 380: '0011000000000101', 434: '0011000000000110', 365: '00110000000001110', 405: '00110000000001111', 426: '00110000000010', 331: '00110000000011', 287: '001100000001', 273: '00110000001', 277: '001100000100', 303: '00110000010100', 340: '00110000010101', 297: '0011000001011', 422: '0011000001100', 356: '001100000110100', 435: '0011000001101010', 316: '0011000001101011', 282: '00110000011011', 413: '001100000111', 322: '00110000100', 274: '0011000010100', 315: '001100001010100', 352: '001100001010101', 292: '00110000101011', 326: '001100001011', 408: '00110000110000000', 404: '00110000110000001', 310: '0011000011000001', 388: '0011000011000010', 378: '0011000011000011', 298: '00110000110001', 333: '001100001100100', 362: '001100001100101', 296: '00110000110011', 272: '001100001101', 276: '001100001110', 415: '001100001111', 121: '00110001', 462: '0011001', 32: '001101', 83: '00111000', 73: '00111001', 84: '00111010', 118: '00111011', 451: '001111', 68: '01000000', 40: '01000001', 184: '01000010', 465: '01000011', 190: '01000100', 188: '01000101', 64: '0100011', 105: '0100100', 61: '01001010', 58: '01001011', 259: '010011000', 319: '0100110010', 369: '01001100110000000', 411: '01001100110000001', 376: '0100110011000001', 300: '010011001100001', 410: '01001100110001000', 400: '01001100110001001', 363: '0100110011000101', 403: '01001100110001100', 399: '01001100110001101', 445: '01001100110001110', 406: '01001100110001111', 311: '0100110011001', 421: '0100110011010', 301: '0100110011011', 472: '010011001110', 358: '010011001111000', 344: '010011001111001', 294: '01001100111101', 280: '0100110011111', 45: '01001101', 111: '0100111', 6: '01010000', 69: '01010001', 115: '0101001', 461: '0101010', 7: '01010110', 59: '01010111', 48: '0101100', 263: '010110100', 267: '01011010100', 383: '01011010101', 359: '0101101011000000', 444: '01011010110000010', 414: '010110101100000110', 377: '010110101100000111', 427: '010110101100001', 428: '010110101100010', 407: '01011010110001100', 443: '01011010110001101', 302: '0101101011000111', 270: '0101101011001', 307: '010110101101000', 387: '0101101011010010', 441: '01011010110100110', 397: '01011010110100111', 278: '01011010110101', 285: '0101101011011', 330: '0101101011100', 328: '0101101011101', 374: '0101101011110000', 398: '01011010111100010', 442: '01011010111100011', 318: '010110101111001', 432: '0101101011110100', 433: '0101101011110101', 304: '010110101111011', 289: '0101101011111', 120: '01011011', 16: '0101110', 67: '01011110', 53: '01011111', 448: '011000', 116: '0110010', 417: '011001100', 253: '011001101', 55: '01100111', 57: '01101000', 54: '01101001', 98: '01101010', 243: '011010110', 254: '011010111', 66: '01101100', 52: '01101101', 112: '0110111', 3: '01110000', 217: '011100010', 419: '011100011000', 364: '011100011001000', 354: '011100011001001', 396: '01110001100101000', 402: '01110001100101001', 314: '01110001100101010', 440: '01110001100101011', 299: '011100011001011', 334: '0111000110011', 262: '01110001101', 269: '01110001110', 447: '011100011110', 381: '011100011111', 218: '011100100', 229: '011100101', 221: '011100110', 209: '011100111', 97: '0111010', 460: '0111011', 210: '011110000', 214: '011110001', 207: '011110010', 245: '011110011', 1: '0111101', 219: '011111000', 250: '011111001', 244: '011111010', 222: '011111011', 102: '01111110', 242: '011111110', 235: '011111111', 8: '10000000', 227: '100000010', 237: '100000011', 284: '10000010000', 418: '10000010001', 468: '1000001001', 223: '100000101', 143: '100000110', 251: '100000111', 103: '10000100', 249: '100001010', 159: '100001011', 457: '1000011', 212: '100010000', 213: '100010001', 247: '100010010', 198: '100010011', 51: '10001010', 246: '100010110', 215: '100010111', 62: '10001100', 220: '100011010', 171: '100011011', 231: '100011100', 155: '100011101', 135: '100011110', 161: '100011111', 456: '1001000', 50: '10010010', 226: '100100110', 252: '100100111', 75: '100101000', 230: '100101001', 10: '10010101', 458: '1001011', 233: '100110000', 126: '100110001', 149: '100110010', 203: '100110011', 173: '100110100', 165: '100110101', 123: '100110110', 151: '100110111', 166: '100111000', 94: '100111001', 134: '100111010', 169: '100111011', 141: '100111100', 234: '100111101', 147: '100111110', 158: '100111111', 157: '101000000', 142: '101000001', 125: '101000010', 162: '101000011', 211: '101000100', 148: '101000101', 130: '101000110', 145: '101000111', 167: '101001000', 164: '101001001', 60: '10100101', 205: '101001100', 170: '101001101', 197: '101001110', 133: '101001111', 449: '101010', 146: '101011000', 201: '101011001', 225: '101011010', 238: '101011011', 236: '101011100', 175: '101011101', 163: '101011110', 90: '101011111', 459: '1011000', 47: '10110010', 5: '10110011', 4: '10110100', 174: '101101010', 140: '101101011', 87: '101101100', 150: '101101101', 138: '101101110', 329: '101101111000000', 286: '101101111000001', 338: '10110111100001', 313: '101101111000100', 395: '10110111100010100', 349: '10110111100010101', 355: '1011011110001011', 325: '10110111100011', 291: '10110111100100', 350: '101101111001010', 391: '10110111100101100', 394: '10110111100101101', 353: '10110111100101110', 379: '10110111100101111', 385: '10110111100110', 348: '101101111001110', 370: '1011011110011110', 312: '1011011110011111', 470: '10110111101', 424: '10110111110000', 423: '10110111110001', 293: '10110111110010', 438: '10110111110011000', 439: '10110111110011001', 351: '1011011111001101', 309: '101101111100111', 275: '1011011111010', 281: '1011011111011', 279: '101101111110', 335: '101101111111000', 372: '1011011111110010', 446: '101101111111001100', 412: '101101111111001101', 382: '10110111111100111', 305: '101101111111010', 431: '1011011111110110', 368: '1011011111110111', 295: '10110111111110', 336: '10110111111111', 101: '1011100', 202: '101110100', 154: '101110101', 239: '101110110', 39: '101110111', 228: '101111000', 153: '101111001', 156: '101111010', 37: '101111011', 241: '101111100', 204: '101111101', 137: '101111110', 132: '101111111', 139: '110000000', 74: '110000001', 56: '11000001', 89: '110000100', 466: '110000101', 93: '110000110', 124: '110000111', 455: '1100010', 81: '110001100', 179: '110001101', 2: '11000111', 464: '11001000', 195: '110010010', 31: '110010011', 208: '11001010', 199: '110010110', 206: '110010111', 29: '110011000', 172: '110011001', 117: '11001101', 131: '110011100', 196: '110011101', 181: '110011110', 129: '110011111', 92: '110100000', 122: '110100001', 144: '11010001', 96: '11010010', 224: '11010011', 23: '110101000', 26: '110101001', 160: '11010101', 49: '11010110', 43: '110101110', 25: '110101111', 30: '110110000', 35: '110110001', 38: '110110010', 27: '110110011', 109: '11011010', 91: '110110110', 185: '110110111', 183: '110111000', 42: '110111001', 416: '11011101', 240: '11011110', 216: '110111110', 178: '110111111', 182: '111000000', 86: '111000001', 104: '11100001', 268: '11100010000', 265: '11100010001', 324: '1110001001000', 346: '111000100100100', 393: '11100010010010100', 345: '11100010010010101', 343: '1110001001001011', 332: '11100010010011', 323: '1110001001010', 342: '111000100101100', 357: '11100010010110100', 436: '11100010010110101', 337: '1110001001011011', 425: '111000100101110', 430: '111000100101111', 266: '111000100110', 283: '1110001001110', 386: '1110001001111000', 347: '1110001001111001', 389: '11100010011110100', 373: '111000100111101010', 401: '111000100111101011', 437: '11100010011110110', 375: '11100010011110111', 360: '1110001001111100', 308: '1110001001111101', 371: '11100010011111100', 392: '11100010011111101', 429: '1110001001111111', 21: '111000101', 191: '11100011', 177: '111001000', 194: '111001001', 46: '11100101', 180: '111001100', 36: '111001101', 65: '11100111', 28: '111010000', 20: '111010001', 18: '111010010', 186: '111010011', 454: '1110101', 256: '1110110', 113: '111011100', 261: '111011101', 44: '111011110', 15: '111011111', 100: '11110000', 19: '111100010', 79: '111100011', 176: '11110010', 128: '11110011', 77: '111101000', 168: '111101001', 71: '111101010', 106: '111101011', 127: '111101100', 136: '111101101', 260: '11110111', 80: '11111000', 78: '111110010', 13: '111110011', 41: '111110100', 200: '111110101', 14: '111110110', 232: '111110111', 192: '11111100', 189: '111111010', 17: '111111011', 463: '11111110', 107: '111111110', 467: '1111111110', 321: '1111111111000', 288: '11111111110010', 290: '11111111110011', 471: '111111111101', 264: '11111111111'}
# 2 blk
# static_tree = {41: '00000000', 76: '00000001', 152: '00000010', 85: '00000011', 105: '0000010', 111: '0000011', 17: '00001000', 488: '0000100100', 321: '000010010100', 425: '00001001010100', 343: '000010010101010', 345: '0000100101010110', 389: '0000100101010111', 288: '0000100101011', 322: '00001001011', 360: '000010011000000', 436: '0000100110000010', 371: '0000100110000011', 317: '00001001100001', 327: '0000100110001', 277: '000010011001', 491: '00001001101', 276: '000010011100', 272: '000010011101', 308: '000010011110000', 339: '000010011110001', 366: '000010011110010', 357: '0000100111100110', 437: '0000100111100111', 429: '000010011110100', 373: '00001001111010100', 401: '00001001111010101', 392: '0000100111101011', 296: '00001001111011', 292: '00001001111100', 390: '0000100111110100', 306: '0000100111110101', 367: '0000100111110110', 341: '0000100111110111', 283: '0000100111111', 34: '00001010', 232: '00001011', 107: '00001100', 82: '00001101', 193: '00001110', 187: '00001111', 465: '0001000', 11: '00010010', 189: '00010011', 457: '0001010', 115: '0001011', 70: '00011000', 248: '00011001', 9: '00011010', 88: '00011011', 12: '00011100', 33: '00011101', 95: '00011110', 478: '00011111', 456: '0010000', 64: '0010001', 464: '0010010', 24: '00100110', 72: '00100111', 463: '0010100', 458: '0010101', 116: '0010110', 259: '001011100', 262: '00101110100', 267: '00101110101', 326: '001011101100', 494: '001011101101', 422: '0010111011100', 282: '00101110111010', 331: '00101110111011', 270: '0010111011110', 380: '0010111011111000', 365: '00101110111110010', 361: '00101110111110011', 352: '001011101111101', 303: '00101110111111', 121: '00101111', 68: '00110000', 118: '00110001', 119: '00110010', 184: '00110011', 73: '00110100', 84: '00110101', 97: '0011011', 462: '0011100', 40: '00111010', 7: '00111011', 1: '0011110', 83: '00111110', 61: '00111111', 319: '0100000000', 487: '0100000001', 483: '010000001', 477: '01000001', 16: '0100001', 461: '0100010', 455: '0100011', 48: '0100100', 416: '01001010', 58: '01001011', 6: '01001100', 188: '01001101', 190: '01001110', 67: '01001111', 460: '0101000', 45: '01010010', 69: '01010011', 449: '010101', 459: '0101100', 59: '01011010', 120: '01011011', 476: '01011100', 66: '01011101', 101: '0101111', 112: '0110000', 297: '0110001000000', 280: '0110001000001', 333: '011000100001000', 300: '011000100001001', 315: '011000100001010', 362: '011000100001011', 435: '0110001000011000', 409: '01100010000110010', 405: '01100010000110011', 356: '011000100001101', 424: '01100010000111', 413: '011000100010', 301: '0110001000110', 497: '0110001000111', 490: '01100010010', 334: '0110001001100', 434: '0110001001101000', 316: '0110001001101001', 378: '0110001001101010', 388: '0110001001101011', 278: '01100010011011', 421: '0110001001110', 328: '0110001001111', 482: '011000101', 53: '01100011', 3: '01100100', 55: '01100101', 54: '01100110', 475: '01100111', 57: '01101000', 103: '01101001', 454: '0110101', 47: '01101100', 62: '01101101', 418: '01101110000', 415: '011011100010', 358: '011011100011000', 310: '0110111000110010', 369: '01101110001100110', 404: '01101110001100111', 427: '011011100011010', 364: '011011100011011', 363: '0110111000111000', 376: '0110111000111001', 359: '0110111000111010', 408: '01101110001110110', 445: '01101110001110111', 374: '0110111000111100', 387: '0110111000111101', 428: '011011100011111', 486: '0110111001', 284: '01101110100', 493: '011011101010', 419: '011011101011', 417: '0110111011', 98: '01101111', 52: '01110000', 8: '01110001', 5: '01110010', 260: '01110011', 257: '01110100', 474: '01110101', 51: '01110110', 253: '011101110', 258: '011101111', 60: '01111000', 102: '01111001', 256: '0111101', 4: '01111100', 10: '01111101', 50: '01111110', 473: '01111111', 383: '10000000000', 411: '10000000001000000', 410: '10000000001000001', 400: '10000000001000010', 403: '10000000001000011', 307: '100000000010001', 294: '10000000001001', 285: '1000000000101', 330: '1000000000110', 344: '100000000011100', 354: '100000000011101', 399: '10000000001111000', 444: '10000000001111001', 314: '10000000001111010', 414: '100000000011110110', 446: '100000000011110111', 432: '1000000000111110', 406: '10000000001111110', 443: '10000000001111111', 263: '1000000001', 217: '100000001', 254: '100000010', 243: '100000011', 453: '1000001', 218: '100001000', 221: '100001001', 219: '100001010', 229: '100001011', 2: '10000110', 250: '100001110', 213: '100001111', 209: '100010000', 261: '100010001', 249: '100010010', 207: '100010011', 214: '100010100', 279: '100010101000', 266: '100010101001', 302: '1000101010100000', 441: '10001010101000010', 407: '10001010101000011', 299: '100010101010001', 318: '100010101010010', 304: '100010101010011', 289: '1000101010101', 447: '100010101011', 489: '10001010110', 269: '10001010111', 472: '10001011', 245: '100011000', 235: '100011001', 210: '100011010', 227: '100011011', 117: '10001110', 251: '100011110', 222: '100011111', 247: '100100000', 481: '100100001', 244: '100100010', 215: '100100011', 237: '100100100', 242: '100100101', 223: '100100110', 220: '100100111', 109: '10010100', 159: '100101010', 155: '100101011', 143: '100101100', 212: '100101101', 231: '100101110', 246: '100101111', 135: '100110000', 252: '100110001', 191: '10011001', 56: '10011010', 123: '100110110', 126: '100110111', 46: '10011100', 167: '100111010', 233: '100111011', 471: '10011110', 171: '100111110', 198: '100111111', 49: '10100000', 161: '101000010', 134: '101000011', 203: '101000100', 423: '10100010100000', 433: '1010001010000100', 442: '10100010100001010', 397: '10100010100001011', 430: '101000101000011', 385: '10100010100010', 325: '10100010100011', 274: '10100010100100', 398: '10100010100101000', 440: '10100010100101001', 370: '1010001010010101', 286: '101000101001011', 338: '10100010100110', 350: '101000101001110', 396: '10100010100111100', 402: '10100010100111101', 355: '1010001010011111', 496: '1010001010100', 329: '101000101010100', 313: '101000101010101', 290: '10100010101011', 381: '101000101011', 485: '1010001011', 169: '101000110', 226: '101000111', 144: '10100100', 125: '101001010', 75: '101001011', 452: '1010011', 157: '101010000', 230: '101010001', 162: '101010010', 94: '101010011', 234: '101010100', 166: '101010101', 104: '10101011', 149: '101011000', 173: '101011001', 151: '101011010', 148: '101011011', 163: '101011100', 236: '101011101', 165: '101011110', 142: '101011111', 175: '101100000', 199: '101100001', 158: '101100010', 133: '101100011', 225: '101100100', 170: '101100101', 238: '101100110', 141: '101100111', 239: '101101000', 164: '101101001', 140: '101101010', 205: '101101011', 451: '1011011', 138: '101110000', 130: '101110001', 204: '101110010', 90: '101110011', 63: '10111010', 470: '10111011', 156: '101111000', 211: '101111001', 201: '101111010', 146: '101111011', 450: '1011111', 197: '110000000', 265: '11000000100', 271: '11000000101', 268: '11000000110', 264: '11000000111', 145: '110000010', 147: '110000011', 153: '110000100', 174: '110000101', 137: '110000110', 202: '110000111', 208: '11000100', 228: '110001010', 172: '110001011', 154: '110001100', 87: '110001101', 224: '11000111', 93: '110010000', 139: '110010001', 132: '110010010', 179: '110010011', 39: '110010100', 241: '110010101', 92: '110010110', 150: '110010111', 96: '11001100', 74: '110011010', 37: '110011011', 89: '110011100', 124: '110011101', 160: '11001111', 81: '110100000', 195: '110100001', 469: '11010001', 181: '110100100', 480: '110100101', 110: '11010011', 100: '11010100', 275: '1101010100000', 348: '110101010000100', 395: '11010101000010100', 391: '11010101000010101', 312: '1101010100001011', 293: '11010101000011', 287: '1101010100010', 336: '11010101000110', 309: '110101010001110', 372: '1101010100011110', 353: '11010101000111110', 379: '11010101000111111', 320: '11010101001', 281: '1101010101000', 324: '1101010101001', 384: '110101010101', 492: '110101010110', 291: '11010101011100', 340: '110101010111010', 298: '110101010111011', 305: '110101010111100', 439: '11010101011110100', 394: '11010101011110101', 349: '11010101011110110', 438: '11010101011110111', 335: '110101010111110', 351: '1101010101111110', 431: '1101010101111111', 29: '110101011', 129: '110101100', 206: '110101101', 65: '11010111', 131: '110110000', 122: '110110001', 240: '11011001', 31: '110110100', 196: '110110101', 23: '110110110', 177: '110110111', 91: '110111000', 43: '110111001', 26: '110111010', 178: '110111011', 27: '110111100', 25: '110111101', 0: '11011111', 468: '11100000', 35: '111000010', 113: '111000011', 99: '11100010', 38: '111000110', 42: '111000111', 30: '111001000', 185: '111001001', 28: '111001010', 182: '111001011', 114: '11100110', 216: '111001110', 183: '111001111', 108: '11101000', 176: '11101001', 86: '111010100', 180: '111010101', 128: '11101011', 194: '111011000', 127: '111011001', 21: '111011010', 36: '111011011', 71: '111011100', 44: '111011101', 467: '11101111', 295: '11110000000000', 346: '111100000000010', 342: '111100000000011', 420: '1111000000001', 368: '1111000000010000', 386: '1111000000010001', 426: '111100000001001', 332: '11110000000101', 323: '1111000000011', 273: '111100000010', 495: '1111000000110', 311: '11110000001110', 377: '111100000011110000', 412: '111100000011110001', 382: '11110000001111001', 347: '1111000000111101', 393: '11110000001111100', 375: '11110000001111101', 337: '1111000000111111', 484: '1111000001', 18: '111100001', 77: '111100010', 106: '111100011', 186: '111100100', 79: '111100101', 19: '111100110', 15: '111100111', 255: '111101000', 20: '111101001', 80: '11110101', 448: '1111011', 192: '11111000', 136: '111110010', 168: '111110011', 78: '111110100', 479: '111110101', 466: '11111011', 32: '1111110', 22: '111111100', 13: '111111101', 14: '111111110', 200: '111111111'}
# 3 blk
# static_tree = {484: '00000000', 17: '00000001', 97: '0000001', 458: '0000010', 295: '0000011000000', 425: '00000110000010', 349: '0000011000001100', 394: '0000011000001101', 347: '000001100000111', 327: '0000011000010', 283: '0000011000011', 508: '00000110001', 503: '0000011001', 277: '000001101000', 343: '000001101001000', 377: '00000110100100100', 412: '00000110100100101', 393: '0000011010010011', 337: '000001101001010', 360: '000001101001011', 288: '0000011010011', 272: '000001101010', 317: '00000110101100', 389: '0000011010110100', 345: '0000011010110101', 366: '000001101011011', 422: '0000011010111', 262: '00000110110', 267: '00000110111', 34: '00000111', 200: '00001000', 483: '00001001', 463: '0000101', 152: '00001100', 85: '00001101', 82: '00001110', 11: '00001111', 64: '0001000', 1: '0001001', 449: '000101', 482: '00011000', 9: '00011001', 107: '00011010', 193: '00011011', 187: '00011100', 481: '00011101', 70: '00011110', 232: '00011111', 462: '0010000', 95: '00100010', 248: '00100011', 16: '0010010', 189: '00100110', 12: '00100111', 33: '00101000', 480: '00101001', 88: '00101010', 259: '001010110', 326: '001010111000', 282: '00101011100100', 308: '001010111001010', 339: '001010111001011', 296: '00101011100110', 292: '00101011100111', 371: '0010101110100000', 436: '0010101110100001', 429: '001010111010001', 331: '00101011101001', 270: '0010101110101', 392: '0010101110110000', 357: '0010101110110001', 298: '001010111011001', 367: '0010101110110100', 437: '0010101110110101', 378: '0010101110110110', 373: '00101011101101110', 401: '00101011101101111', 301: '0010101110111', 502: '0010101111', 461: '0010110', 48: '0010111', 479: '00110000', 72: '00110001', 478: '00110010', 121: '00110011', 101: '0011010', 454: '0011011', 460: '0011100', 68: '00111010', 24: '00111011', 477: '00111100', 84: '00111101', 257: '00111110', 119: '00111111', 118: '01000000', 73: '01000001', 459: '0100001', 496: '010001000', 319: '0100010010', 507: '01000100110', 280: '0100010011100', 362: '010001001110100', 306: '0100010011101010', 390: '0100010011101011', 358: '010001001110110', 352: '010001001110111', 334: '0100010011110', 424: '01000100111110', 380: '0100010011111100', 341: '0100010011111101', 333: '010001001111111', 83: '01000101', 7: '01000110', 184: '01000111', 476: '01001000', 67: '01001001', 40: '01001010', 61: '01001011', 188: '01001100', 45: '01001101', 69: '01001110', 58: '01001111', 6: '01010000', 475: '01010001', 260: '01010010', 59: '01010011', 112: '0101010', 256: '0101011', 120: '01011000', 190: '01011001', 453: '0101101', 474: '01011100', 495: '010111010', 418: '01011101100', 328: '0101110110100', 297: '0101110110101', 303: '01011101101100', 300: '010111011011010', 364: '010111011011011', 421: '0101110110111', 417: '0101110111', 66: '01011110', 47: '01011111', 53: '01100000', 473: '01100001', 54: '01100010', 3: '01100011', 55: '01100100', 5: '01100101', 103: '01100110', 258: '011001110', 501: '0110011110', 316: '0110011111000000', 365: '01100111110000010', 361: '01100111110000011', 356: '011001111100001', 315: '011001111100010', 363: '0110011111000110', 435: '0110011111000111', 374: '0110011111001000', 388: '0110011111001001', 434: '0110011111001010', 310: '0110011111001011', 278: '01100111110011', 413: '011001111101', 279: '011001111110', 419: '011001111111', 57: '01101000', 52: '01101001', 472: '01101010', 98: '01101011', 452: '0110110', 494: '011011100', 263: '0110111010', 506: '01101110110', 266: '011011101110', 409: '01101110111100000', 369: '01101110111100001', 376: '0110111011110001', 427: '011011101111001', 359: '0110111011110100', 405: '01101110111101010', 445: '01101110111101011', 354: '011011101111011', 285: '0110111011111', 8: '01101111', 51: '01110000', 62: '01110001', 102: '01110010', 50: '01110011', 4: '01110100', 60: '01110101', 471: '01110110', 261: '011101110', 493: '011101111', 10: '01111000', 415: '011110010000', 330: '0111100100010', 387: '0111100100011000', 404: '01111001000110010', 408: '01111001000110011', 428: '011110010001101', 423: '01111001000111', 284: '01111001001', 383: '01111001010', 510: '011110010110', 344: '011110010111000', 348: '011110010111001', 385: '01111001011101', 336: '01111001011110', 302: '0111100101111100', 314: '01111001011111010', 411: '01111001011111011', 307: '011110010111111', 253: '011110011', 450: '0111101', 451: '0111110', 117: '01111110', 470: '01111111', 2: '10000000', 109: '10000001', 46: '10000010', 49: '10000011', 500: '1000010000', 320: '10000100010', 290: '10000100011000', 325: '10000100011001', 294: '10000100011010', 274: '10000100011011', 430: '100001000111000', 410: '10000100011100100', 400: '10000100011100101', 443: '10000100011100110', 403: '10000100011100111', 338: '10000100011101', 287: '1000010001111', 254: '100001001', 56: '10000101', 469: '10000110', 243: '100001110', 217: '100001111', 191: '10001000', 221: '100010010', 218: '100010011', 104: '10001010', 219: '100010110', 492: '100010111', 213: '100011000', 229: '100011001', 209: '100011010', 250: '100011011', 110: '10001110', 271: '10001111000', 265: '10001111001', 264: '10001111010', 414: '100011110110000000', 446: '100011110110000001', 399: '10001111011000001', 370: '1000111101100001', 350: '100011110110001', 299: '100011110110010', 432: '1000111101100110', 444: '10001111011001110', 442: '10001111011001111', 289: '1000111101101', 384: '100011110111', 245: '100011111', 249: '100100000', 207: '100100001', 63: '10010001', 227: '100100100', 210: '100100101', 251: '100100110', 214: '100100111', 247: '100101000', 235: '100101001', 0: '10010101', 244: '100101100', 222: '100101101', 215: '100101110', 252: '100101111', 223: '100110000', 242: '100110001', 144: '10011001', 237: '100110100', 220: '100110101', 468: '10011011', 268: '10011100000', 505: '10011100001', 269: '10011100010', 447: '100111000110', 304: '100111000111000', 441: '10011100011100100', 406: '10011100011100101', 433: '1001110001110011', 318: '100111000111010', 372: '1001110001110110', 407: '10011100011101110', 440: '10011100011101111', 275: '1001110001111', 246: '100111001', 212: '100111010', 155: '100111011', 100: '10011110', 114: '10011111', 159: '101000000', 143: '101000001', 208: '10100001', 135: '101000100', 123: '101000101', 161: '101000110', 231: '101000111', 126: '101001000', 167: '101001001', 224: '10100101', 233: '101001100', 134: '101001101', 125: '101001110', 225: '101001111', 160: '10101000', 171: '101010010', 198: '101010011', 162: '101010100', 491: '101010101', 203: '101010110', 226: '101010111', 75: '101011000', 94: '101011001', 148: '101011010', 157: '101011011', 199: '101011100', 133: '101011101', 163: '101011110', 238: '101011111', 108: '10110000', 149: '101100010', 175: '101100011', 169: '101100100', 166: '101100101', 165: '101100110', 230: '101100111', 96: '10110100', 99: '10110101', 211: '101101100', 499: '1011011010', 355: '1011011011000000', 397: '10110110110000010', 398: '10110110110000011', 329: '101101101100001', 313: '101101101100010', 335: '101101101100011', 324: '1011011011001', 509: '101101101101', 286: '101101101110000', 312: '1011011011100010', 396: '10110110111000110', 395: '10110110111000111', 340: '101101101110010', 402: '10110110111001100', 382: '10110110111001101', 368: '1011011011100111', 420: '1011011011101', 323: '1011011011110', 311: '10110110111110', 332: '10110110111111', 234: '101101110', 142: '101101111', 197: '101110000', 236: '101110001', 65: '10111001', 173: '101110100', 239: '101110101', 158: '101110110', 151: '101110111', 164: '101111000', 130: '101111001', 146: '101111010', 138: '101111011', 240: '10111110', 205: '101111110', 137: '101111111', 141: '110000000', 170: '110000001', 156: '110000010', 90: '110000011', 140: '110000100', 145: '110000101', 467: '11000011', 201: '110001000', 204: '110001001', 147: '110001010', 228: '110001011', 174: '110001100', 153: '110001101', 93: '110001110', 172: '110001111', 132: '110010000', 37: '110010001', 202: '110010010', 92: '110010011', 179: '110010100', 154: '110010101', 139: '110010110', 124: '110010111', 87: '110011000', 195: '110011001', 490: '110011010', 74: '110011011', 241: '110011100', 181: '110011101', 176: '11001111', 81: '110100000', 129: '110100001', 131: '110100010', 89: '110100011', 448: '1101001', 39: '110101000', 150: '110101001', 206: '110101010', 29: '110101011', 177: '110101100', 91: '110101101', 466: '11010111', 32: '1101100', 122: '110110100', 196: '110110101', 178: '110110110', 309: '110110111000000', 426: '110110111000001', 293: '11011011100001', 281: '1101101110001', 322: '110110111001', 381: '110110111010', 379: '11011011101100000', 439: '11011011101100001', 351: '1101101110110001', 346: '110110111011001', 291: '11011011101101', 342: '110110111011100', 305: '110110111011101', 391: '11011011101111000', 438: '11011011101111001', 386: '1101101110111101', 375: '11011011101111100', 353: '11011011101111101', 431: '1101101110111111', 498: '1101101111', 23: '110111000', 31: '110111001', 128: '11011101', 489: '110111100', 43: '110111101', 42: '110111110', 38: '110111111', 111: '11100000', 35: '111000010', 26: '111000011', 105: '11100010', 113: '111000110', 30: '111000111', 27: '111001000', 25: '111001001', 185: '111001010', 180: '111001011', 488: '111001100', 28: '111001101', 182: '111001110', 86: '111001111', 183: '111010000', 127: '111010001', 71: '111010010', 194: '111010011', 36: '111010100', 216: '111010101', 192: '11101011', 255: '111011000', 106: '111011001', 465: '11101101', 44: '111011100', 21: '111011101', 115: '11101111', 80: '11110000', 456: '11110001', 457: '11110010', 487: '111100110', 77: '111100111', 18: '111101000', 504: '11110100100', 321: '1111010010100', 276: '1111010010101', 273: '111101001011', 497: '1111010011', 79: '111101010', 19: '111101011', 15: '111101100', 186: '111101101', 78: '111101110', 486: '111101111', 41: '111110000', 20: '111110001', 116: '11111001', 22: '111110100', 14: '111110101', 485: '111110110', 136: '111110111', 464: '11111100', 13: '111111010', 76: '111111011', 168: '111111100', 416: '111111101', 455: '11111111'}

def static_tree_encode(metabyte_chunk: List[MetaByte]) -> str:
    return "".join(static_tree[num.num_encoding] for num in metabyte_chunk)
def static_tree_decode(binary: str) -> List[MetaByte]:
    chunk = []
    idx = 0
    while idx < len(binary):
        for num, code in static_tree.items():
            if binary[idx:].startswith(code):
                idx += len(code)
                chunk.append(decode_value(num))
                break
    return chunk

def compress_all_stages(page, mem_blks, chunksize, stage3=False):
    metabyte_page = pad_input(page)
    metabyte_chunks = chunk_input(metabyte_page, chunksize)

    # comp stage 1:
    stage1_compressed = metabyte_chunks
    # apply csc0
    stage1_compressed = apply_chunkwise(csc0_encode, stage1_compressed)

    # comp stage 2:
    stage2_compressed, dict_bits = bse_encode_9b(stage1_compressed, mem_blks)

    if stage3:
        # comp stage 3
        stage3_compressed = apply_chunkwise(static_tree_encode, stage2_compressed)
        final_compressed = stage3_compressed
    else:
        stage2_compressed_bin = apply_chunkwise(metabyte_page_to_binary, stage2_compressed)
        final_compressed = stage2_compressed_bin

    return final_compressed, dict_bits, stage1_compressed, stage2_compressed

def decompress_all_stages(compressed_chunks_bin, dict_bits, stage3=False):
    global bse_applications
    global csc_applications

    bse_applications = [0] * 63
    csc_applications = [0] * 4
    if stage3:
        stage3_decompressed = apply_chunkwise(static_tree_decode, compressed_chunks_bin)
        stage2_decompressed = apply_chunkwise(bse_decode_9b, stage3_decompressed, dict_bits)
    else:
        # decomp stage 2 :
        stage2_decompressed = apply_chunkwise(binary_to_metabyte_page, compressed_chunks_bin)
        stage2_decompressed = apply_chunkwise(bse_decode_9b, stage2_decompressed, dict_bits)
        
    # decomp stage 1:
    stage1_decompressed = stage2_decompressed
    # reverse csc0
    stage1_decompressed = apply_chunkwise(csc0_decode, stage1_decompressed)

    # unchunk and unpad
    decompressed_page = unchunk_output(stage1_decompressed)

    return decompressed_page

# just 2 MAG blocks and bse to add and we're cooking.


# just rework to make everybody map the alg?
# only bse need to be one dict for all
from test_alg import temp_parallel_test_compression_ratio_chunks
if __name__ == '__main__':
    CHUNK_SIZE = 256 # sets of 4 mem blocks # probably a good balance for now.
    STAGE3 = False
    directory_path = 'well_rounded_tests/*'
    file_paths = glob.glob(directory_path)
    for benchmark in sorted(file_paths):
        pages = read_file_in_chunks(benchmark)
        for mem_block_count in range(2, 3):
            # results = parallel_test_compression_ratio_chunks(
            results = temp_parallel_test_compression_ratio_chunks(
            # results = serial_test_compression_ratio_chunks(
                pages,
                compress_all_stages,
                decompress_all_stages,
                [mem_block_count, CHUNK_SIZE, STAGE3],
                [STAGE3]
            )
            # with open(
            #     f"nine_bit/csc0_results_static_tree.txt", "a"
            # ) as f:
            #     f.write(f"benchmark: {benchmark.split('/')[1]}, mem blocks: {mem_block_count}, chunk size: {CHUNK_SIZE}, {results}\n")
            with open("dict_entries_2blk.txt", 'a') as f:
                f.write(f"{benchmark.split('/')[-1]}\n")
                f.write(f"{results}\n")
        print(benchmark)
