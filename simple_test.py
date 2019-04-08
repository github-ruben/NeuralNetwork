import time
start = time.time()

p = []
for i in range(10000000):
    x = 2*i
    y = 3
    p.append(x*y)
    
end = time.time()
print(end-start)
