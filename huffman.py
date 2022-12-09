import numpy as np
from collections import Counter
import math
class Node:
    def __init__(self, Letter= None, Prob = None, ToNode=None, *,code = ""):
        self.Letter = Letter
        self.Prob = Prob
        self.ToNode = ToNode
        self.code = code

    def Add_1_toCode(self):
        self.code = "1" + self.code
    def Add_0_toCode(self):
        self.code = "0" + self.code


class MasterCell:
    def __init__(self, ToDeleted, middle=None, right=None):
        self.Sum = 0
        self.ToDeleted = ToDeleted
        self.middle = middle
        self.right = right

    def listprint(self):
        printval = self.middle
        while printval is not None:
            print(printval.Letter)
            printval = printval.ToNode


class Huffman:
    def __init__(self):
        self.Head = None
        self.Num = None

    def InitializeAll(self, TotalNums):  # Initiate all MasterCells to make its middle points at its corresponding nodes
        self.Head = MasterCell('1')
        Follow = self.Head
        for i in range(2, TotalNums + 1):
            Follow.right = MasterCell(str(i))
            Follow = Follow.right
        self.Num = TotalNums + 1

    def InitializeAllNodes(self, my_dict):  # this intiate all nodes by its letter and probability
        self.InitializeAll(len(my_dict))
        point = self.Head
        for char in my_dict:
            node = Node(char, my_dict[char])
            point.middle = node
            point.Sum = my_dict[char]
            point = point.right

    def printMasters(self):  # Print Master name(I give it name just to trace it) and its sum
        print(self.Head.ToDeleted, self.Head.Sum)
        pointer = self.Head.right
        while pointer is not None:
            print(pointer.ToDeleted, pointer.Sum)
            pointer = pointer.right

    def printNodeContent(self):  # Print the node content (Letter, Probability)
        print(self.Head.middle.Letter, self.Head.middle.Prob)
        pointer = self.Head.right
        while pointer is not None:
            print(pointer.middle.Letter, pointer.middle.Prob)
            pointer = pointer.right

    def sortedMerge(self, a, b):  # Sort Master Cells by its sum variable
        result = None
        if a == None:
            return b
        if b == None:
            return a
        if a.Sum <= b.Sum:
            result = a
            result.right = self.sortedMerge(a.right, b)
        else:
            result = b
            result.right = self.sortedMerge(a, b.right)
        return result

    def mergeSort(self, h):  # Follower to sort function
        if h == None or h.right == None:
            return h
        middle = self.getMiddle(h)
        nexttomiddle = middle.right
        middle.right = None
        left = self.mergeSort(h)
        right = self.mergeSort(nexttomiddle)
        sortedlist = self.sortedMerge(left, right)
        return sortedlist

    def getMiddle(self, head):  # Follower to sort function
        if (head == None):
            return head
        slow = head
        fast = head
        while (fast.right != None and
               fast.right.right != None):
            slow = slow.right
            fast = fast.right.right
        return slow

    def AppendingP1_P2(self):  # Append a Content of a master cell into the next one(Accumulate nodes,Sum)
        pointer1 = self.Head
        pointer2 = self.Head.right
        point = pointer2.middle
        point.Add_0_toCode()
        while (point.ToNode != None):
            point = point.ToNode
            point.Add_0_toCode()

        point.ToNode = Node(pointer1.middle.Letter, pointer1.middle.Prob, code=pointer1.middle.code)
        # point = point.ToNode
        pointer1Help = pointer1.middle
        point.ToNode.Add_1_toCode()
        while (pointer1Help.ToNode != None):
            pointer1Help = pointer1Help.ToNode
            point = point.ToNode
            point.ToNode = Node(pointer1Help.Letter, pointer1Help.Prob, code=pointer1Help.code)
            point.ToNode.Add_1_toCode()

        pointer2.Sum += pointer1.Sum
        self.Head = pointer2


class ManageHuffman:
    def __init__(self, my_dict):
        self.my_dict = my_dict
        self.Huff = Huffman()
        self.Huff.InitializeAllNodes(self.my_dict)
        self.Huff.Head = self.Huff.mergeSort(self.Huff.Head)
        self.ReturnDict = {}
        self.CodeListDictionary()
        self.Entropy = self.calc_Entropy()
        self.AverageLength = self.calc_AverageLength()
        self.Efficiency = self.Entropy / self.AverageLength

    def printCodeList(self):  # Print each character and its code
        for char in self.ReturnDict:
            print(char, self.ReturnDict[char])

    def CodeListDictionary(self):  # Coding processing happens here(Sort, append, Repeat till right == Null)
        while (self.Huff.Head.right != None):
            self.Huff.Head = self.Huff.mergeSort(self.Huff.Head)
            self.Huff.AppendingP1_P2()

        Master = self.Huff.Head.middle
        self.ReturnDict[Master.Letter] = Master.code
        while (Master.ToNode != None):
            Master = Master.ToNode
            self.ReturnDict[Master.Letter] = Master.code

    # def printDataFrame(self):  # Print all Each Character, probability of that character, and its code in dataframe
    #     AllList = []
    #     for char in self.ReturnDict:
    #         Small_list = []
    #         Small_list.append(char)
    #         Small_list.append(self.my_dict[char])
    #         Small_list.append(self.ReturnDict[char])
    #         AllList.append(Small_list)
    #     Table = pd.DataFrame(AllList, columns=['Character', 'Probability', 'Code'])
    #     return Table

    def calc_Entropy(self):  # Calculating the Entropy
        Sum = 0
        for char in self.my_dict:
            Sum += -1 * (self.my_dict[char] * math.log2(self.my_dict[char]))
        return Sum

    def calc_AverageLength(self):  # Calculating Average Code Length
        Sum = 0
        for char in self.my_dict:
            Sum += self.my_dict[char] * len(self.ReturnDict[char])
        return Sum

    def Summary(self):  # Report summary of all coding Characteristics
        print(self.printDataFrame())
        print('Entropy is: {}'.format(self.Entropy))
        print('Average Code Length is: {}'.format(self.AverageLength))
        print('Efficiency is: {}'.format(self.Efficiency))


    def EncodeList(self, Array):
        List = ""
        Totallength = 0
        for i in range(len(Array)):
            char = Array[i]
            code = self.ReturnDict[char]
            Totallength += len(code)
            List += code
        print('Total length is: {}'.format(Totallength))
        return List,self.ReturnDict

    def DecodeList(self, Code):
        SubCode = ""
        List = []
        for i in range(len(Code)):
            SubCode += Code[i]
            if (SubCode in self.ReturnDict.values()):
                for Letter, code in self.ReturnDict.items():
                    if code == SubCode:
                        List.append(Letter)
                        SubCode = ""
                        pass
        return List


def split_code_chars_probs(comp_code):
    all_keys = list(Counter(comp_code).keys())  # equals to list(set(words))
    all_probs =  np.array(list(Counter(comp_code).values()))   # counts the elements' frequency
    all_probs = all_probs /  len(comp_code)
    d = {x: all_probs[i] for i,x in np.ndenumerate(all_keys) }
    return all_keys, all_probs, d

def encode_huffman(input):
    lets, probs,d  = split_code_chars_probs(input)
    encoded, d = ManageHuffman(d).EncodeList(input)
    print(d)
    return encoded, d

# def encode_huffman(data):
#     codes = run_huffman_algoirthm(data)
#     op = []
#     for c in data:
#         c_code = codes[c]
#         op.append(c_code)
#     output_string = "".join(op)
#     return codes ,output_string


def decode_huffman(coded_data, d):
    i = 0
    all_prev_code = ""
    output = []
    while True and i < len(coded_data):
        current_bit = coded_data[i]
        all_prev_code = all_prev_code + current_bit
        if str(all_prev_code) in d.keys():
            output.append(d[all_prev_code])
            all_prev_code = ""
        i +=1
    return output



