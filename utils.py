
def form_seq_graph(seq):
    graph = {}
    for i, s in enumerate(seq[:-1]):        
        if s not in graph:
            graph[s] = []
        graph[s].append(seq[i+1])     
    return graph

# sample a random last edge graph
def sample_le_graph(graph, last_nt):
    le_graph = {}
    for vx in graph:
        le_graph[vx] = []
        if vx not in last_nt:
            le_graph[vx].append(random.choice(graph[vx]))
    return le_graph        

# check whether there exists an Eulerian walk
# from seq[0] to seq[-1] in the shuffled
# sequence
def check_le_graph(le_graph, last_nt):
    for vx in le_graph:
        if vx not in last_nt:
            if not find_path(le_graph, vx, last_nt):
                return False
    return True            

# function from: http://www.python.org/doc/essays/graphs/
# check whether there is a path between two nodes in a 
# graph
def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
        return None       
        
# generate a new seq graph based on the last edge graph
# while randomly permuting all other edges        
def form_new_graph(graph, le_graph, last_nt):
    new_graph = {}
    for vx in graph:
        new_graph[vx] = []
        temp_edges = graph[vx]
        if vx not in last_nt:
            temp_edges.remove(le_graph[vx][0])
        random.shuffle(temp_edges)        
        for ux in temp_edges:
            new_graph[vx].append(ux)
        if vx not in last_nt:    
            new_graph[vx].append(le_graph[vx][0])
    return new_graph                     
      
# walk through the shuffled graph and make the
# new sequence       
def form_shuffled_seq(new_graph, init_nt, len_seq):
    is_done = False
    new_seq = init_nt
    while not is_done:
        last_nt  = new_seq[-1]
        new_seq += new_graph[last_nt][0]
        new_graph[last_nt].pop(0)
        if len(new_seq) >= len_seq:
            is_done = True
    return new_seq    

# verify the nucl
def verify_counts(seq, shuf_seq):
    kmers = {}
    # Forming the k-mer library
    kmer_range = range(1,3)
    for k in kmer_range:
        for tk in itert.product('ACGTN', repeat=k):
            tkey = ''.join(i for i in tk)
            kmers[tkey] = [0,0]
                        
    kmers[seq[0]][0] = 1
    kmers[shuf_seq[0]][1] = 1      
    for k in kmer_range:
        for l in range(len(seq)-k+1):
            tkey = seq[l:l+k]
            kmers[tkey][0] += 1
            tkey = shuf_seq[l:l+k]
            kmers[tkey][1] += 1
    for tk in kmers:
        if kmers[tk][0] != kmers[tk][1]:
            return False    
    return True

_preprocess_seq = ['N']*256;
_preprocess_seq[ord('a')] = _preprocess_seq[ord('A')] = 'A';  # Map A => A
_preprocess_seq[ord('c')] = _preprocess_seq[ord('C')] = 'C';  # Map C => C
_preprocess_seq[ord('g')] = _preprocess_seq[ord('G')] = 'G';  # Map G => G
_preprocess_seq[ord('t')] = _preprocess_seq[ord('T')] = 'T';  # Map T => T
_preprocess_seq[ord('u')] = _preprocess_seq[ord('U')] = 'T';  # Map U => T
_preprocess_seq = "".join(_preprocess_seq)
            
def preprocess_seq(seq):
    return seq.translate(_preprocess_seq) 

def doublet_shuffle(seq, verify=False):
    seq = preprocess_seq(seq)
    last_nt = seq[-1]
    graph = form_seq_graph(seq)
    # sample a random last edge graph
    is_ok = False
    while not is_ok:
        le_graph = sample_le_graph(graph, last_nt)
        # check the last edge graph
        is_ok = check_le_graph(le_graph, last_nt)
    new_graph = form_new_graph(graph, le_graph, last_nt)
    shuf_seq  = form_shuffled_seq(new_graph, seq[0], len(seq))
    if verify:
        assert(verify_counts(seq, shuf_seq))
    return shuf_seq


def kmer_former(seq, kmerlen):
    kmers = {}
    for j in range(len(seq) - kmerlen + 1):
        tkey = seq[j:j+kmerlen]
        if tkey not in kmers:
            kmers[tkey] = True
    return kmers

def verify_kmer(shuf_seq, kmers, kmerlen):
    for j in range(len(shuf_seq) - kmerlen + 1):
        tkey = shuf_seq[j:j+kmerlen]
        if tkey in kmers:
            return False
    return True
