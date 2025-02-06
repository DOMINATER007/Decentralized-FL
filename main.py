from clients import c1,c2,c3,c4,c5,c6,c7

if __name__ == "__main__":
    clients=[c1.client1,c2.client2,c3.client3,c4.client4,c5.client5,c6.client6,c7.client7]
    locations = [c.coordinates for c in clients]
    print(locations)