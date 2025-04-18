import statistics 


def election_helper(cluster_clients,data,alpha=0.5):
    rank={}
    for i in cluster_clients:
        rank[i]=0
    #print(data)
    for client in cluster_clients:
        temp_score={}

        others_data={i:data[i] for i in cluster_clients if i!=client}
        # print(others_data)
        # print(type(others_data))
        for i,temp_data in others_data.items():
            temp_score[i]=0
            score = (statistics.median(temp_data["accuracy_history_list"]) * 
                (statistics.stdev(temp_data["accuracy_history_list"]) if len(temp_data["accuracy_history_list"]) > 1 else 0.005) ** (-1))

            score+=(temp_data["Model_info"]["hidden_layers"])
            temp_score[i]=score
        mx=0
        mx_id=None
        for k,v in temp_score.items():
            if v>mx:
                mx=v
                mx_id=k
            elif v==mx:
                if others_data[k]["Model_info"]["total_params"]>others_data[mx_id]["Model_info"]["total_params"]:
                    mx_id=k
        rank[mx_id]+=1
        
        
    print(f"\nRank : {rank}\n")   
    mr=0
    mr_id=None
    for k,v in rank.items():
            if v>=mr:
                mr=v
                mr_id=k
    return mr_id
            
        
        
    