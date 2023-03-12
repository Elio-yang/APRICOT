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
UN=0
err_files = []

def process_graph(dir_name):
    
    global UN
    global err_files
    
    new_dir = TRAIN_DATASET_ROOT+"/"+dir_name
    cmd = "mkdir -p "+new_dir
    do_shell_command_call(cmd)
    
    content_name = RAND_TRAIN_DATASET_ROOT+"/"+dir_name+"/"+dir_name+".content"
    link_name = RAND_TRAIN_DATASET_ROOT+"/"+dir_name+"/"+dir_name+".link"
    new_content_name = new_dir+"/"+dir_name+".content"
    new_link_name = new_dir+"/"+dir_name+".link"
    
    try:    
        # store new file
        
        # read in the content file
        with open(content_name, 'r') as f:
            lines = f.readlines()
        # get the total node count
        node_count = int(lines[0])
        # create a dictionary to map old node indices to new node indices
        index_map = {}
        new_index = 0
        new_content = []
        # iterate through the content file and re-index nodes
        for line in lines[1:]:
            # extract the old node index
            
            data = line.strip('\n').strip().split(',')
            old_index = data[0]
            features = data[1:6]
            label = data[6]
            # add the old index to the index map if it's not already there
            if old_index not in index_map:
                index_map[old_index] = new_index
                new_index += 1
            
            new_idx = index_map[old_index]
            data = str(new_idx)+","
            for f in features:
                data+= str(f)+","
            data += label
            data += "\n"
            new_content.append(data)
            
        # create a new content file with re-indexed nodes
        with open(new_content_name, 'w',newline="") as f:
            f.write(str(node_count) + '\n')
            f.writelines(new_content)

        # read in the link file
        with open(link_name, 'r') as f:
            lines = f.readlines()
        # create a new link file with re-indexed nodes
        
        new_link = []
        for line in lines[1:]:
                source, target = line.strip('\n').split('\t')
                #print(dir,source)
                new_source = str(index_map[source])
                new_target = str(index_map[target])
                new_line = new_source + '\t' + new_target + '\n'
                new_link.append(new_line)
        
        with open(new_link_name, 'w') as f:
            f.write(lines[0])
            f.writelines(new_link)
    except KeyError as e:
        print(dir_name)
        cmd = "rm -rf "+ new_dir
        do_shell_command_call(cmd)
        UN+=1
        err_log = dir_name+"@"+ str(e)
        err_files.append(err_log)
        print(traceback.format_exc())
    except Exception:
        print(traceback.format_exc())
        


if __name__=="__main__":
    

    
    with cd(RAND_TRAIN_DATASET_ROOT):
        graph_dirs = do_shell_command("ls")
        #print(gjson_files)
        # deal with single file from an benchmark
    
    # create dir
    td_path = TRAIN_DATASET_ROOT
    do_shell_command_call("mkdir -p "+td_path)

    PROCESS_POOL = Pool(8)    
    process_bar = tqdm(total=len(graph_dirs))
    
    def update(*a):
        process_bar.update()
        
    #random.shuffle(gjson_files)
    for dir in graph_dirs:
        #filepath = SEP_GJSON_ROOT+"/"+file        
        PROCESS_POOL.apply_async(process_graph,args=(dir,),callback=update)    
        #process_graph(dir)    
    
    PROCESS_POOL.close()
    PROCESS_POOL.join()
    
    print("end")
    
    with open("reorder_err.log","w") as f:
        f.write(str(UN))
        f.writelines(err_files)
