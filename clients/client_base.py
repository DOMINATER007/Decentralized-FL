class client(object):
    def __init__(self,model,coordinates,cluster,accuracy_hist,Lfreq,start_time,end_time):
        self.model=model
        self.coordinates=coordinates
        self.cluster=cluster
        self.accuracy_hist=accuracy_hist
        self.Lfreq=Lfreq
        self.start_time=start_time
        self.end_time=end_time

