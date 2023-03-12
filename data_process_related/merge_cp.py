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



def process_gjson(file):
    #print("hi")
    with open(file,"r") as fp:
        #---------------------------------------------------------------------
        gjson=json.load(fp)
        idx=0
        all_func_idx=set()
        func_idx={}
        def_func_map={}
        for function in gjson['graph']['function']:
            #print(function['name'])
            # TABLE 1: FUNCTION_NAME IDX
            # We should only record defined function
            all_func_idx.add(idx)
            func_idx[idx]=function['name']
            idx+=1
        #---------------------------------------------------------------------
        # @function that is declared
        # without instruction infomation
        # INT
        del_func_idx=set()
        
        for node in gjson['nodes']:
            if node['text']=="; undefined function":
                del_func_idx.add(node['function'])      
        
        # @function that is defined
        # with instruction infomation
        def_func_idx = all_func_idx-del_func_idx
        
        for idx in def_func_idx:
            # func_name -> id
            def_func_map[func_idx[idx]]=idx
        
        #print(def_func_map)
        #---------------------------------------------------------------------
        # MAKE THE DATASET
        conten_=""
        link_=""
        br_cnt_in_gra={}
        
        type_OUTSIDE=[]
        type_BR=[]
        type_BR_pre=[]
        type_DIRECT=[]
        node_cnt =0
        for node in gjson['nodes']:
            node_cnt+=1 
            
            """
                del_func_idx
                def_func_idx:   w_func_idx
                                nw_func_idx
                
            """
            
            # texture part
            
            has_weight = False
            true_weight = 0
            false_weight = 0
            tot_weight = 0
            branch_type = 0
            
            text_ir=""
            if node.__contains__('features'):
                
                text_ir=node['features']['full_text'][0]
                if text_ir=="":# undefined func here
                    text_ir=node['text']
                    if not text_ir=="; undefined function":
                        print("OOV1")
                        text_ir="OOV"
                #print(text_ir)
                
                # simple deal with weight here
                if node['features'].__contains__('llvm_profile_true_weight'):
                    has_weight=True
                    true_weight = node['features']['llvm_profile_true_weight'][0]
                    false_weight = node['features']['llvm_profile_false_weight'][0]
                    tot_weight = node['features']['llvm_profile_total_weight'][0]
                    
                    branch_type = float(true_weight/tot_weight)
            else:
                # only [externel]
                text_ir=node['text']
                if not text_ir=="[external]":
                    print("OOV2")
                    text_ir="OOV"
            
            #print(text_ir)
            
            function_id = node['function']
            instr_idx = node['id']
            flow_type = node['type']
            opcode = node['text']
            # remain consideration
            feature = np.random.randint(0,2,1433)
            #print(feature)
            strs=""
            for fv in np.nditer(feature):
                    strs+=str(fv)
                    strs+=","
            #print(text_ir)
            # assign type OUTSIDE (3)
            if function_id in del_func_idx:
                data = str(instr_idx) +","+ strs + "OUTSIDE"+"\n"
                type_OUTSIDE.append(data)
            # assign type <LEFT,MID,RIGHT>
            elif has_weight==True:
                data = str(instr_idx)+","+ strs
                
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
            # assign type DIRECT (4)
            else :
                data = str(instr_idx)+","+ strs + "DIRECT"+"\n"
                type_DIRECT.append(data)
    #---------------------------------------------------------------------
        #print(br_cnt_pgo)
        #print(br_cnt_in_gra)  
        #print(len(br_cnt_pgo))
        #print(len(br_cnt_in_gra))
        # make sure they got the same branch
    
    seqhash = hashlib.md5()
    seqhash.update(file.encode('utf-8'))
    res = seqhash.hexdigest()
    
    
    sample_path = td_path+"/"+res+"/"
    cmds="mkdir -p "+ sample_path
    do_shell_command_call(cmds)
    
    c_path = sample_path+"/"+res+".content"
    l_path = sample_path+"/"+res+".link"
    
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
    
    return True

if __name__=="__main__":
    with cd(GJSON_ROOT):
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
        filepath = GJSON_ROOT+"/"+file        
        PROCESS_POOL.apply_async(process_gjson,args=(filepath,),callback=update)    
        #process_gjson(filepath)    
    
    PROCESS_POOL.close()
    PROCESS_POOL.join()
    
    # with cd(HOME):
    #     list_file = "graph_list.txt"
    #     with open(list_file,"w") as fp:
    #         fp.writelines(all_files)   
    print("end")
