with open("gpu.txt", "r") as fin:
  max_gpu = -1
  for line in fin:
    words = line.split()
    if words[1] == 'MiB':
      max_gpu = max(max_gpu, int(words[0]))
  print(max_gpu)    
