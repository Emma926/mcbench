'''
TensorFlow Tuner

Improve TensorFlow model's performance on CPU. This code is under Apache 2.0 Lisence.
More details about this tuner please refer to the following paper. If you find it 
useful, please cite our paper.

Wang, Yu Emma, Carole-Jean Wu, Xiaodong Wang, Kim Hazelwood, and David Brooks. 
"Exploiting Parallelism Opportunities with Deep Learning Frameworks." 
arXiv preprint arXiv:1908.04705 (2019).

This code was tested with fc_tf.py and inception.

Yu Emma Wang
10/16/2019
'''

# To simplify the model graph, this function finds
# the parent heavy ops of a given op.
def findparentop(curr, g_dict, heavy_ops):
  l = []
  node = []
  seen = set()
  for i in g_dict[curr]:
    node.append(i)

  while node:
    n = node[0]
    del node[0]
    seen.add(n)
    if n in heavy_ops:
      l.append(n)
      continue
    for i in g_dict[n]:
      if not i in seen:
        node.append(i)
  return list(set(l))

# find the nodes at the bottom of the graph
# such nodes are not parents of any nodes
def find_bottom_node(s_graph, heavy_ops):
  p_set = set()
  for op in heavy_ops:
    for p in s_graph[op]:
      p_set.add(p)
  return set(heavy_ops) - p_set

# find the depth of the graph by depth first search
def dfs(n, s_graph):
  if len(s_graph[n]) == 0:
    return 1
  depths = []
  for p in s_graph[n]:
    depths.append(dfs(p, s_graph))
  return max(depths) + 1

# An heavy operator is defined to be the ops taking 
# much more execution time than other ops.
# Such ops include MatMul, Conv and Embedding ops.
def isheavy(n):
  if 'gradient' in n:
    return False
  if 'MatMul' in n.split('/')[-1] or 'Conv' in n.split('/')[-1]:
    return True
  return False

# the interface of TF-Tuner.
def tftuner(graph):
  # Initialize data structures
  g_dict = {}
  s_graph = {}
  heavy_ops = []
  for op in graph.get_operations():
    g_dict[op.name] = []
    for i in op.inputs:
      g_dict[op.name].append(i.name.split(':')[0])
      #if isheavy(op.name):
      #  print(op.name, '<-', i)
    if isheavy(op.name):
      heavy_ops.append(op.name)
  for op in heavy_ops:
    s_graph[op] = []
  
  print('=========== Graph Summary ===========')
  total_nodes = 0
  for op in g_dict:
    total_nodes += len(g_dict[op])
  print('Total Ops:', len(g_dict), 'Total nodes:', total_nodes)
  print('Heavy Ops:', len(heavy_ops))

  print('=========== Heavy Ops ===========')
  for op in s_graph.keys():
      print(op)
  print('# of heavy ops:', len(heavy_ops))
  #exit()

  print('=========== Simplify Graph ===========')
  for op in reversed(heavy_ops):
    l = findparentop(op, g_dict, heavy_ops)
    s_graph[op] = l
    print(op, l) 

  print('=========== Find Graph Depth ===========')
  bottoms = find_bottom_node(s_graph, heavy_ops)
  print('Bottom nodes: ', bottoms)
  depths = []
  for node in bottoms:
    d = dfs(node, s_graph)
    depths.append(d)

  heavy_op = len(s_graph)
  heavy_layer = max(depths) 
  avg_width = heavy_op*1.0/heavy_layer
  print('*** Heavy Ops = ', heavy_op)
  print('*** Layers = ', heavy_layer)
  print('*** Avg Graph Width = ', heavy_op/heavy_layer)
  return avg_width 
