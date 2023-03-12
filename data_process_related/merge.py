import os
import subprocess
import re
from helper import *
import json
import numpy as np
import hashlib
import threading
import random
from multiprocessing.pool import Pool
import multiprocessing
import traceback
from tqdm import tqdm

# 1. IR->idx
# 2. idx|feature|label
# 3. save to .content

# from programl json file to reconstruct file IR 
# Solution1: label as the type this instruction goes T in next
#       [IDX][INST]<FEATURE>[LABEL]
#       [IDX][IDX]
# Solution2: label as the type this edge been executed from idx1->idx2
#       [IDX][INST]<FEATURE>
#       [IDX1][IDX2][LABEL]

def process_gjson(file,indentity):
    try:
        code_emb_set=set()
        #print("hi")
        with open(file,"r") as fp:
            #---------------------------------------------------------------------
            gjson=json.load(fp)
            
            type_OUTSIDE=[]
            type_BR=[]
            type_DIRECT=[]
            
            func_id = gjson['function_id']
            
            for node in gjson['nodes']:
                has_weight = False
                true_weight = 0
                false_weight = 0
                tot_weight = 0
                branch_type = 0.0
                
                br = 0
                out = 0
                direct = 0
                
                
                # make sure all nodes have "features"
                # simple deal with weight here
                if node['features'].__contains__('llvm_profile_true_weight'):
                    has_weight=True
                    true_weight = node['features']['llvm_profile_true_weight'][0]
                    false_weight = node['features']['llvm_profile_false_weight'][0]
                    tot_weight = node['features']['llvm_profile_total_weight'][0]
                    branch_type = float(true_weight/tot_weight)
                    # donate this is a branch
                else:
                    branch_type = -1.0
                
                #print(text_ir)
                
                function_id = node['function']
                instr_idx = node['id']
                node_type = node['type']
                opcode = node['text']
                
                
                # <br,outside,dir,type,inst2vec>
                
                
                
                inst2vec_emb = node['features']['inst2vec_embedding'][0]
                todb = opcode+" "+str(inst2vec_emb)+"\n"
                code_emb_set.add(todb)
                
                
                # remain consideration
                feature = inst2vec_emb
                #print(feature)
                strs= str(node_type)+","+str(feature)+","
                
                # assign type <LEFT,MID,RIGHT>
                if has_weight==True:
                    out = 0
                    br = 1
                    direct = 0
                    feature_vec = str(br)+","+str(out)+","+str(direct)+","
                    data = str(instr_idx)+","+feature_vec+strs
                    # type 0, branch prob <= 0.2
                    # tyoe 1, branch prob >0.8
                    # type 2, branch prob in (0.2,0.8]
                    if branch_type <= 0.20:
                        data += "LESS" + "\n"
                    elif branch_type >0.80:
                        data += "MORE" + "\n"
                    else:
                        data += "MID" +"\n"
                    type_BR.append(data)
                # assign type OUTSIDE (3)
                elif func_id != function_id:
                    out = 1
                    br = 0
                    direct = 0
                    feature_vec = str(br)+","+str(out)+","+str(direct)+","
                    data = str(instr_idx) +","+ feature_vec + strs + "OUTSIDE"+"\n"
                    type_OUTSIDE.append(data)
                # assign type DIRECT (4)
                else:
                    out = 0
                    br = 0
                    direct = 1
                    feature_vec = str(br)+","+str(out)+","+str(direct)+","
                    data = str(instr_idx)+","+ feature_vec + strs + "DIRECT"+"\n"
                    type_DIRECT.append(data)
        #---------------------------------------------------------------------
        # seqhash = hashlib.md5()
        # seqhash.update(file.encode('utf-8'))
        # res = seqhash.hexdigest()
        res = indentity
        
        
        sample_path = td_path+"/"+res+"/"
        cmds="mkdir -p "+ sample_path
        do_shell_command_call(cmds)
        
        c_path = sample_path+"/"+res+".content"
        l_path = sample_path+"/"+res+".link"
        m_path = MAP_ROOT+"/"+res+".map"
        
        V=len(type_DIRECT) + len(type_OUTSIDE) + len(type_BR)
        type_OUTSIDE = [str(V) + "\n"] + type_OUTSIDE
        
        with open(c_path,"w",newline="") as rfp:
            rfp.writelines(type_OUTSIDE)
            rfp.writelines(type_DIRECT)
            rfp.writelines(type_BR)

        links=gjson['links']
        edges=[]
        for edge in links:
            s=edge['source']
            e=edge['target']
            info = str(s)+"\t"+str(e)+"\n"
            edges.append(info)
        
        edges = [str(len(edges))+"\n"] + edges
        with open(l_path,"w",newline="") as lfp:
            # the first line is the number of edges
            lfp.writelines(edges)
        
        with open(m_path,"w",newline="") as mfp:
            mfp.writelines(code_emb_set)
        
        return True
    except Exception:
        print(traceback.format_exc())
        return False
        

if __name__=="__main__":
    with cd(SEP_GJSON_ROOT):
        gjson_files = do_shell_command("ls")
        #print(gjson_files)
        # deal with single file from an benchmark
    
    # create dir
    td_path = RAND_TRAIN_DATASET_ROOT
    do_shell_command_call("mkdir -p "+td_path)

    PROCESS_POOL = Pool(8)    
    process_bar = tqdm(total=len(gjson_files))
    def update(*a):
        process_bar.update()
        
    #random.shuffle(gjson_files)
    for file in gjson_files:
        filepath = SEP_GJSON_ROOT+"/"+file        
        PROCESS_POOL.apply_async(process_gjson,args=(filepath,file,),callback=update)    
        #process_gjson(filepath)    
    
    PROCESS_POOL.close()
    PROCESS_POOL.join()
    
    # with cd(HOME):
    #     list_file = "graph_list.txt"
    #     with open(list_file,"w") as fp:
    #         fp.writelines(all_files)   
    print("end")
