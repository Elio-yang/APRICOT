import re
import programl as pg
from helper import *
import json
from multiprocessing.pool import Pool
import multiprocessing

from programl.ir.llvm.inst2vec_encoder import Inst2vecEncoder

from tqdm import tqdm

def process_pir(pir,pirpath,enc,inst2vec):
    graph_json,flag=ir2gra(pirpath,pversion="13",inst2vec_encoder=enc,inst2vec=inst2vec)
    if flag==True:
        save_path=GJSON_ROOT+"/"+pir+".json"
        save_json(save_path,graph_json) 
        return True
    return False


if __name__=="__main__":
    with cd(PIR_ROOT):
        pirs=do_shell_command("ls")
        #print(pirs)

    PROCESS_POOL = Pool(POOL_SIZE)
    
    encoder = Inst2vecEncoder()
    inst2vec = True
    
    process_bar = tqdm(total=len(pirs))
    def update(*a):
        process_bar.update()
        
    for pir in pirs:
        pirpth=PIR_ROOT+"/"+pir
        PROCESS_POOL.apply_async(process_pir,args=(pir,pirpth,encoder,inst2vec,),callback=update)
    
    PROCESS_POOL.close()
    PROCESS_POOL.join()
