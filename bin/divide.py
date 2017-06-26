# coding: utf-8
data = open("wiki_data", "r").read()
data = data.split()
size = len(data)

training = int(size*0.8)
validation = int(size*0.1)
testing = int(size*0.1)

with open("wiki.train.txt", "w") as f:
    f.write(data[0:training])
    
with open("wiki.train.txt", "w") as f:
    f.write(" ".join(data[0:training]))
    
with open("wiki.valid.txt", "w") as f:
    f.write(" ".join(data[training+1:training+1+validation]))
    
with open("wiki.test.txt", "w") as f:
    f.write(" ".join(data[training+1+validation+1:-1]))
    
