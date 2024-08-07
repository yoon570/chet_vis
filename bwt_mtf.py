# fast(ish) decode from Donald Adjeroh, Timothy Bell, Amar Mukherjee book
from huffman import *
from small_bse import *
from good_csc_zero import *
from csc_create_escape import csc_encode
from tqdm import tqdm
import csv
import argparse

def bwt_encode(data):
    rotations = sorted(data[i:]+data[:i] for i in range(len(data)))
    return list(list(zip(*rotations))[-1]), rotations.index(data)

def bwt_decode(L, a):
    n = len(L)
    K = [0]*256
    M = [0]*256
    C = [0]*n
    for idx, num in enumerate(L):
        C[idx] = K[num]
        K[num] += 1
    total = 0
    for c in range(256):
        M[c] = total
        total += K[c]
    i = a
    Q = [0]*n
    for j in reversed(range(n)):
        Q[j] = L[i]
        i = C[i] + M[L[i]]
    return Q

def mtf_encode(data):
    byte_values = list(range(256))
    transformed = []
    for num in data:
        list_index = byte_values.index(num)
        transformed.append(list_index)
        byte_values.insert(0, byte_values.pop(list_index))
    return transformed

def mtf_decode(transformed):
    byte_values = list(range(256))
    data = []
    for num in transformed:
        data.append(byte_values[num])
        byte_values.insert(0, byte_values.pop(num))
    return data

PAGE_LEN = 4096

if __name__ == '__main__':    
    with open("data.csv", "w", newline="") as datafile:
        benchmarks = [
            "declipseparsed_3",
            "parsec_splash2x.water_spatial5dump_3",
            "sparkbench_LogisticRegression5dump_3",
            "xalanparsed_3"
        ]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--pages", required=True)
        
        args = parser.parse_args()
        
        statistics = ""
        page_num = int(args.pages)
        
        statwriter = csv.writer(datafile, delimiter=",")
        statwriter.writerow(["Page", "BSE len", "HUF len", "MTF len", "BWT len", "BSE cr", "HUF cr", "MTF cr", "BWT cr", "MTF worse?", "BWT worse?"])
        
        for benchmark in benchmarks:
            pages = read_file_in_chunks(f"labeled_dumps/{benchmark}.bin")
            progress_bar = tqdm(total=page_num, desc=f"{benchmark}", ncols = 100)
            
            bas_compsum = 0
            huf_compsum = 0
            mtf_compsum = 0
            bwt_compsum = 0
            
            for p_idx, page in enumerate(pages):
                
                if p_idx >= page_num:
                    progress_bar.close()
                    break
                
                # BSE --> Huffman
                post_csc = csc_encode(page)
                post_bse = bse_encode(post_csc, 3)
                post_huf = huffman_encode(post_bse, 15)
                
                # BSE --> MTF --> Huffman
                post_mtf = mtf_encode(post_bse)
                post_mtf_huf = huffman_encode(post_mtf, 15)
                
                # BSE --> BWT --> MTF --> Huffman
                bwt_pair = bwt_encode(post_bse)
                bwt_metadata = bwt_pair[1]
                post_bwt = bwt_pair[0]
                
                post_bwt_mtf = mtf_encode(post_bwt)
                post_bwt_mtf_huf = huffman_encode(post_bwt_mtf, 15)
                
                bas_pagelen = len(post_bse)
                huf_pagelen = len(post_huf)
                mtf_pagelen = len(post_mtf_huf)
                bwt_pagelen = len(post_bwt_mtf_huf) + 8
                
                bas_cr = PAGE_LEN / len(post_bse)
                huf_cr = PAGE_LEN / len(post_huf)
                mtf_cr = PAGE_LEN / len(post_mtf_huf)
                bwt_cr = PAGE_LEN / len(post_bwt_mtf_huf)
                
                bas_compsum += bas_pagelen
                huf_compsum += huf_pagelen
                mtf_compsum += mtf_pagelen
                bwt_compsum += bwt_pagelen

                progress_bar.update(1)
                
                huf_worse = ""
                mtf_worse = ""
                bwt_worse = ""
                
                if bas_pagelen < huf_pagelen: huf_worse = "x"
                if bas_pagelen < mtf_pagelen: mtf_worse = "x"
                if bas_pagelen < bwt_pagelen: bwt_worse = "x"
                    
                statwriter.writerow([p_idx, bas_pagelen, huf_pagelen, mtf_pagelen, bwt_pagelen, bas_cr, huf_cr, mtf_cr, bwt_cr])    
            
            uncomp_sum = PAGE_LEN * page_num
            print(f"Compression ratios = BSE: {uncomp_sum / bas_compsum}, HUF: {uncomp_sum / huf_compsum}, MTF: {uncomp_sum / mtf_compsum}, BWT: {uncomp_sum / bwt_compsum}")      
                
            progress_bar.close()