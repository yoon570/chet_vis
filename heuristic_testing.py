from small_bse import *
from good_csc import *
from csc_create_escape import csc_encode
import argparse
from tqdm import tqdm

if __name__ == '__main__':    
    with open("data.csv", "w", newline="") as datafile:
        benchmarks = [
            "dacapo_avrora",
            "spec_xz"
        ]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--pages", required=True)
        args = parser.parse_args()
        
        page_num = int(args.pages)
        
        for benchmark in benchmarks:
            pages = read_file_in_chunks(f"well_rounded_tests/{benchmark}_rand1k")
            progress_bar = tqdm(total=page_num, desc=f"{benchmark}", ncols = 70)
            
            heu2_tot = 0
            heu1_tot = 0
            
            for p_idx, page in enumerate(pages):
                
                if p_idx >= page_num:
                    progress_bar.close()
                    break
                
                # Heuristic sqrt
                post_csc2 = csc_encode(page, 2)
                post_bse2 = bse_encode(post_csc2, 2)
                
                # Heuristic savings/span
                post_csc1 = csc_encode(page, 1)
                post_bse1 = bse_encode(post_csc1, 2)
                
                heu2_tot += len(post_bse2)
                heu1_tot += len(post_bse1)
                
                progress_bar.update(1)
                
            progress_bar.close()
            
            print(f"Stats = heu_quot_cr: {(4096 * page_num) / heu1_tot}, heu_sqrt_tot: {(4096 * page_num) / heu2_tot}")