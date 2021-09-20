import torch
from fairseq.models.bart import BARTModel
import os
from tqdm import tqdm
import argparse
from os.path import join
import logging
import files2rouge
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm
import os
import time

def generate_TLDRs(max_len,checkpoint_file='checkpoint_1_5000.pt',checkpoint_dir="checkpoints"):
    folder='data'
    hypo_dir="hypo/"
    print(checkpoint_file)
    bsz=32

    bart = BARTModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=folder+'-bin',
        batch_size=bsz,
        max_tokens=1024*bsz,
    )

    bart.cuda()
    bart.eval()
    bart.half()
    import time
    for split_fold in ['test','dev']:
        with open(folder+'/'+split_fold+'.source', encoding="utf-8") as source, open(hypo_dir+'/'+checkpoint_file+'.'+split_fold+'.hypo', 'w', encoding="utf-8") as fout:
            # sline =
            slines = []
            # N=1000
            # head = [next(source) for x in range(N)]
            start_time = time.time()
            icount = 0
            count = 0
            # bsz = 1

            for sline in tqdm(source):
                # icount+=1
                # if icount == 2:
                #     continue
                if count !=0 and count % bsz == 0:
                    with torch.no_grad():
                        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=max_len, min_len=20, no_repeat_ngram_size=3)

                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                        # print(hypothesis)
                    slines = []

                slines.append(sline.strip())
                count += 1

            if slines != []:
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=max_len, min_len=20, no_repeat_ngram_size=3)
                print(hypotheses_batch)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
            print("--- %s seconds"+str(max_len)+" ---" + str( (time.time() - start_time)))

def maybe_percentages(r, percentages):
    if percentages:
        for r_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for m_type in ['f', 'p', 'r']:
                x = r[r_type][m_type]
                r[r_type][m_type] = x * 100
    return r
import sys
if __name__ == '__main__':
    start = time.time()
    generate_TLDRs(200,checkpoint_file=sys.argv[1])
    end = time.time()
    print(f'Time to run script: {(end - start)} sec')