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

if __name__ == '__main__':    
    with open("yoon_segmented.txt.", "r") as datafile:
        bm_hufavg = 0
        bm_mtfavg = 0
        bm_bwtavg = 0
        while True:
            bm_title = datafile.readline()
            pages_parse = datafile.readline()
            if not bm_title:
                break
        
            pages = eval(pages_parse)
            
            benchmark_baseline = 0
            benchmark_huf = 0
            benchmark_mtf = 0
            benchmark_bwt = 0
            
            for page in pages:
                new_page_mtf = []
                new_page_bwt = []
                
                for chunk in page:
                    new_chunk = []
                    
                    for value in chunk:
                        new_val = value % 256
                        new_chunk.append(new_val)
                        
                    post_mtf = mtf_encode(new_chunk)
                    new_page_mtf.append(post_mtf)
                    
                    post_bwt = bwt_encode(new_chunk)
                    post_mtf = mtf_encode(post_bwt[0])   
                    new_page_bwt.append(post_mtf)                 
                
                page_measurement = [x for subx in page for x in subx]
                newpage_mtf_measurement = [y for suby in new_page_mtf for y in suby]
                newpage_bwt_measurement = [z for subz in new_page_bwt for z in subz]
                newpage_bwt = huffman_encode(newpage_bwt_measurement, 15)
                newpage_mtf = huffman_encode(newpage_mtf_measurement, 15)
                newpage_huf = huffman_encode(page_measurement, 15)
                
                benchmark_baseline += len(page_measurement)
                
                if len(newpage_huf) < len(page_measurement):
                    benchmark_huf += len(newpage_huf)
                else:
                    benchmark_huf += len(page_measurement)
                    
                if len(newpage_mtf) < len(page_measurement):
                    benchmark_mtf += len(newpage_mtf)
                else:
                    benchmark_mtf += len(page_measurement)
                    
                if len(newpage_bwt) < len(page_measurement):
                    benchmark_bwt += len(newpage_bwt)
                else:
                    benchmark_bwt += len(page_measurement)
            
            print(bm_title + "---------")
            print(f"Huf: {benchmark_baseline / benchmark_huf}, MTF: {benchmark_baseline / benchmark_mtf}, BWT: {benchmark_baseline / benchmark_bwt}")
            bm_hufavg += benchmark_baseline / benchmark_huf
            bm_mtfavg += benchmark_baseline / benchmark_mtf
            bm_bwtavg += benchmark_baseline / benchmark_bwt
        
        print("Overall average\n--------------\n")
        print(f"Huf: {bm_hufavg / 31}, MTF: {bm_mtfavg / 31}, BWT: {bm_bwtavg / 31}")