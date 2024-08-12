from huffman import *
from small_bse import *
from good_csc import *
from csc_create_escape import csc_encode
from tqdm import tqdm
import argparse

PAGE_LEN = 4096

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

def chunk_page(page, chunk_size):
    # Split the byte object into chunks of the specified size
    return [page[i:i+chunk_size] for i in range(0, len(page), chunk_size)]

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
        parser.add_argument("-cs", "--chunksize", required=True)
        
        args = parser.parse_args()
        
        statistics = ""
        page_num = int(args.pages)
        chunk_size = int(args.chunksize)
        
        for benchmark in benchmarks:
            pages = read_file_in_chunks(f"labeled_dumps/{benchmark}.bin")
            progress_bar = tqdm(total=page_num, desc=f"{benchmark}", ncols = 70)
            
            for p_idx, page in enumerate(pages):
                
                if p_idx >= page_num:
                    progress_bar.close()
                    break
                
                # BSE --> Huffman
                post_csc = csc_encode(page, 2)
                post_bse = bse_encode(post_csc, 3)
                
                # BSE --> MTF --> Huffman                
                bse_int = bytes([x for x in post_bse])
                print(bse_int)
                bse_chunks = chunk_page(bse_int, chunk_size)
                mtf_chunks = []
                for chunk in bse_chunks:
                    mtf_chunks.append(mtf_encode(chunk))
                
                post_mtf = [item for sublist in mtf_chunks for item in sublist]
                
                
                assert(pre_mtf == post_bse)

                progress_bar.update(1)
                
            progress_bar.close()