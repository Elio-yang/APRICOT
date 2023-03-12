import os
import subprocess
import programl as pg
import json

RAND_TRAIN_DATASET_ROOT="/home/jasony/Desktop/GNN4CO/rand_train_data"
TRAIN_DATASET_ROOT="/home/jasony/Desktop/GNN4CO/train_data"
GJSON_ROOT="/home/jasony/Desktop/GNN4CO/GJSON"
SEP_GJSON_ROOT="/home/jasony/Desktop/GNN4CO/SEP_GJSON"
PIR_ROOT="/home/jasony/Desktop/GNN4CO/PIR"
#PIR_ROOT="/home/jasony/Desktop/GNN4CO/ProGraML/for_test"
MAP_ROOT="/home/jasony/Desktop/GNN4CO/MAP"
HOME= "/home/jasony/Desktop/GNN4CO"
TEST_ROOT = "/home/jasony/Desktop/GNN4CO/TEST"
POOL_SIZE = 8

def simple_ir_graph(SIMPLE_IR,version="10"):
    return pg.from_llvm_ir(
        SIMPLE_IR,
        version=version,
    )


def ir2gra(filename,pversion="10",inst2vec_encoder=None,inst2vec=False):
    with open(filename,"r") as fp:
        IR_Text=fp.read()
        try:
            g=simple_ir_graph(IR_Text,version=pversion)
            #print(type(g).__name__)  
            if inst2vec:
                g=inst2vec_encoder.Encode(g)
            graph_json=json.dumps(pg.to_json(g), indent=2)
        except:
            print("Can't deal this file")
            print(filename)
            return None,False
        else:
            return graph_json,True

def save_json(filename,mjson):
    with open(filename,"w") as fp:
        fp.write(mjson)

# enter dir
class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        
def do_shell_command(cmd):
    command=(str(cmd))
    print (command)
    #subprocess.call(command, shell=True)
    output=subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
    return [str.decode().strip("\n") for str in output.stdout.readlines()]
def do_shell_command_call(cmd):
    command=(str(cmd))
    #print (command)
    subprocess.call(command, shell=True)

