import statistics 


def election_helper(cluster_clients,data,alpha=0.5):
    rank={}
    for i in cluster_clients:
        rank[i]=0
    for client in cluster_clients:
        temp_score={}
        others_data={i:data[i] for i in cluster_clients if i!=client}
        print(others_data)
        print(type(others_data))
        for i,temp_data in others_data.items():
            temp_score[i]=0
            score=alpha*(statistics.median(temp_data["accuracy_history_list"])*((statistics.stdev(temp_data["accuracy_history_list"])+0.005)**(-1)))
            score+=(1-alpha)*temp_data["History"]
            temp_score[i]=score
        mx=0
        mx_id=None
        for k,v in temp_score.items():
            if v>=mx:
                mx=v
                mx_id=k
        rank[mx_id]+=1
        
    mr=0
    mr_id=None
    for k,v in rank.items():
            if v>=mr:
                mr=v
                mr_id=k
    return mr_id
            
        
        
    