import numpy as np
import cv2
import matplotlib.pyplot as plt
import heapq 

# IMAGE UTILS
def im2gray(image: np.array):
    # Input: image -> np.array [3 channels]
    # Output: img -> np.array [1 channel]
    img = np.array(image[:,:,0]*0.2989 + image[:,:,1]*0.587 + image[:,:,2]*0.114, dtype = int)
    return img

def read_img(FILEPATH: str):
    # Input: FILEPATH -> str ["path/to/image.png"]
    # Output: image -> np.array [Greyscale]
    image = cv2.imread(FILEPATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = im2gray(image)
    return image


# GETTING PROBS
def get_freqs(image: np.array):
    # Input: image -> np.array [Greyscale]
    # Output: sorted_dict -> dict [keys: 0-255, freqs]
    prob_array = {}

    for j in range(256):
        prob_array[j] = (np.sum(image == j))
        
    sorted_list = sorted(prob_array.items(), key=lambda x:x[1], reverse = True)

    sorted_dict = {z[0]: z[1] for z in sorted_list}

    return sorted_dict

# NODE CLASS
class node: 
    def __init__(self, freq, symbol, left=None, right=None): 
        self.freq = freq 
        self.symbol = symbol 
        self.left = left 
        self.right = right 
        self.huff = '' 
  
    def __lt__(self, nxt): 
        return self.freq < nxt.freq 

    
# HUFFMAN - THE HEART AND SOUL
def huffman(prob_array: dict):
    # Input: prob_array -> dict [keys: 0-255, freqs]
    # Output: nodes -> heap of nodes
    nodes = []
    chars = list(prob_array.keys())
    freq = list(prob_array.values())
    
    for x in range(len(chars)): 
        heapq.heappush(nodes, node(freq[x], chars[x])) 

    while len(nodes) > 1: 
        left = heapq.heappop(nodes) 
        right = heapq.heappop(nodes) 

        left.huff = 0
        right.huff = 1

        newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right) 
        heapq.heappush(nodes, newNode) 
        
    return nodes


# ON TO COMMUNICATION BLOCKS
def transmitter(image, coding_scheme, IM_SHAPE_X, IM_SHAPE_Y):
    # Inputs: {image -> np.array [Greyscale],
    #          coding_scheme -> dict,
    #          IM_SHAPE_X -> image.shape[0],
    #          IM_SHAPE_Y -> image.shape[1]}
    # Output: transmitted -> str
    transmitted = ""

    for iter_x in range(IM_SHAPE_X):
        for iter_y in range(IM_SHAPE_Y):
            transmitted += coding_scheme[image[iter_x, iter_y]]
    return transmitted

def decoder(transmitted, IM_SHAPE_X, IM_SHAPE_Y):
    # Inputs: {transmitted -> str,
    #          IM_SHAPE_X -> image.shape[0],
    #          IM_SHAPE_Y -> image.shape[1],
    #          coding_scheme -> dict}
    # Output: transmitted -> str
    global coding_scheme
    recovered_img = np.zeros((IM_SHAPE_Y, IM_SHAPE_X))
    decoding_scheme = {v: k for k, v in coding_scheme.items()}
    
    img_ctr = 0
    ptr = 0

    while ptr < len(transmitted):
        len_c = 1
        while transmitted[ptr:ptr+len_c] not in decoding_scheme.keys():
            len_c += 1


        code = transmitted[ptr:ptr+len_c]

        val = decoding_scheme[code]

        recovered_img[img_ctr%IM_SHAPE_Y, img_ctr//IM_SHAPE_Y] = val

        img_ctr += 1
        ptr += len_c
        
        
    return recovered_img.T

coding_scheme = {}
def printNodes(node, val=''): 
    global coding_scheme
    newVal = val + str(node.huff) 

    if(node.left): 
        printNodes(node.left, newVal) 
    if(node.right): 
        printNodes(node.right, newVal) 

    if(not node.left and not node.right): 
        coding_scheme[node.symbol] = newVal
    return

def huffmanenco(image):
    global coding_scheme
    IM_SHAPE_X, IM_SHAPE_Y = image.shape[:2]
    freqs = get_freqs(image)
    nodes = huffman(freqs)
    printNodes(nodes[0])
    transmitted = transmitter(image, coding_scheme, IM_SHAPE_X, IM_SHAPE_Y)
    return transmitted
