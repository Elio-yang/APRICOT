import os
import json
from helper import *
from multiprocessing.pool import Pool
from tqdm import tqdm
import traceback

def separate_json_by_function(filename):
    try:
        with open(filename) as file:
            json_data = json.load(file)
        # create a map to hold the graph for each function
        function_graphs = {}

        # iterate through each node and assign it to the corresponding function's graph
        for node in json_data["nodes"]:
            function_index = node["function"]
            
            if function_index not in function_graphs:
                # create a new graph for this function if it doesn't exist yet
                function_graphs[function_index] = {
                    "directed": True,
                    "multigraph": False,
                    "has_branch": False, 
                    "function_id":function_index,
                    "nodes": [],
                    "links": [],
                }
            
            if function_graphs[function_index]["has_branch"] == False:
                if node.__contains__('features'):
                    if node['features'].__contains__('llvm_profile_true_weight'):
                        function_graphs[function_index]["has_branch"]=True
            # Not modifying the node structure
            # therefore, inst2vec is included without extra processing
            function_graphs[function_index]["nodes"].append(node)
        
        # {id1:id2}
        # add node[id2] into function id1's nodes
        OUTSIDE_MAP=set()
        
        # iterate through each link and assign it to the corresponding function's graph
        for link in json_data["links"]:
            
            # index
            source_node = link["source"]
            target_node = link["target"]
            
            #if source_node == 112588 and target_node == 104179:
            #    print("debug_encounter")
            
            # TODO: when adding outside nodes, there might be dup.
            
            # nodes are indexed 
            # which function the source or target belongs to
            # ala. s or t in current func, add both
            source_function_index = json_data["nodes"][source_node]["function"]
            target_function_index = json_data["nodes"][target_node]["function"]
            
            # if this link is related to 2 non-br functions
            # dont deal them
            if function_graphs[source_function_index]["has_branch"] == False and function_graphs[target_function_index]["has_branch"] == False:
                continue
            
            # this edge is in the same function
            if source_function_index == target_function_index:
                # if both nodes are in the same function, add the link to that function's graph
                function_graphs[source_function_index]["links"].append(link)
            elif source_function_index in function_graphs and target_function_index in function_graphs:
                # add outside target
                # cause function_idx is a larger scope
                # nodes could be dup. but link will not
                pair1=str(source_function_index)+":"+str(target_node)
                if pair1 not in OUTSIDE_MAP:
                    function_graphs[source_function_index]["nodes"].append(json_data["nodes"][target_node])
                    OUTSIDE_MAP.add(pair1)
                function_graphs[source_function_index]["links"].append(link)
                
                # add outside source
                pair2=str(target_function_index)+":"+str(source_node)
                if pair2 not in OUTSIDE_MAP:
                    function_graphs[target_function_index]["nodes"].append(json_data["nodes"][source_node])
                    OUTSIDE_MAP.add(pair2)
                function_graphs[target_function_index]["links"].append(link)

        # convert the function graphs back to JSON format and return them
        function_jsons = []
        for function_index, function_graph in function_graphs.items():
                # only store those br-functions
                if function_graphs[function_index]["has_branch"]:
                    function_jsons.append(json.dumps(function_graph))
        return function_jsons
    except Exception:
        print(traceback.format_exc())
        return None

input_dir= GJSON_ROOT
output_dir = SEP_GJSON_ROOT

def process(filename):
    # Load the JSON file into a dictionary
    jsonname = input_dir+"/"+filename
    jsons=separate_json_by_function(jsonname)
    if jsons is not None:
        for idx,fj in enumerate(jsons):
            output_filename = SEP_GJSON_ROOT+"/"+filename[:-5] + "_" + str(idx) + ".json"
            with open(output_filename, "w") as file:
                file.write(fj)

if __name__=="__main__":


    
    PROCESS_POOL = Pool(POOL_SIZE)
    
    
    works = os.listdir(input_dir)
    
    process_bar = tqdm(total=len(works))
    def update(*a):
        process_bar.update()
    
    # Loop through all JSON files in the input directory
    for filename in works:
        if filename.endswith(".json"):
            PROCESS_POOL.apply_async(process,args=(filename,),callback=update)
    
    PROCESS_POOL.close()
    PROCESS_POOL.join()