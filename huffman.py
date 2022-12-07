import numpy as np
from collections import Counter
class Element(object):
  """
    A single element which encapsulate a letter assciated with its probability, and code
  """
  def __init__(self, prob, let = "", code=""):
      """
        Probability of such letter
        code which will be computed
        letter for printing only
      """
      self.prob = prob
      self.code = code
      self.let = let

  def append_c(self, c):
    """
      Append 0, or 1 to the start of the code as the it propagates
    """
    self.code = c + self.code

  def __gt__(self, other):
    """
      To decide wheter current object is greater than other object or not
      it depends on if the other is jus and element ==> then it compares probabilty
      if othe is list of elements ==> compares current probability with sum of probabilites of the other list
    """
    if isinstance(other, Element):
        if self.prob >= other.prob:
            return True
        return False
    if isinstance(other, ListElements):
        if self.prob >= other.get_sum_probs():
            return True
        return False

  def __str__(self):
      return "Letter: {}, Probability: {:.4f}, Code: {}".format(self.let,self.prob, self.code)

  def get_sum_probs(self):
      """
        Just returns the probability
      """
      return self.prob

  def __len__(self):
      return 1

class ListElements(object):
  """
    List that can hold all elements, or list of elements
  """

  def __init__(self, elements):
      """
          elements: List[] of Element objects
      """
      self.lst = []
      for ele in elements:
          self.lst.append(ele)

  def get_sum_probs(self):
      """
      returns sum of probabilites in the list, if the list contains list, it loops over the inner list as well
      """
      sum_probs = 0
      for ele in self.lst:
          sum_probs += ele.get_sum_probs()
      return sum_probs

  def __str__(self):
      s = "Type: ListElements [ "
      for ele in self.lst:
          s += ele.__str__() + ", "
      s += "]"
      return s

  def __gt__(self, other):
      """
        compares sum of probabiltyes, it uses "get_sum_probs" which exists in both Element and ListElement,
        and depeneding on type of class, it decides which one it runs
      """
      if self.get_sum_probs() > other.get_sum_probs():
          return True
      return False

  def sort_me(self):
      """
         Sort the list besed on probabilites
      """
      self.lst = sorted(self.lst, reverse=True)
      return self

  def __len__(self):
      return len(self.lst)

  def get_orig_len(self):
      """
        return length of length without any modification
      """
      orig_len = 0
      for ele in self.lst:
          orig_len += len(ele)
      return orig_len

  def __getitem__(self, index):
      if index < 0:
          return self.lst[len(self.lst) - index]
      return self.lst[index]

  def trunc_last(self):
      """
        encapsulate last two elements in just one list and append that list in the right most of the original list
      """
      new_list = ListElements([self.lst[-1], self.lst[-2]])
      self.lst = self.lst[:-1]
      self.lst[-1] = new_list

  def append_all(self, c):
      """
        for earch element, which can be "Element" or ListElement" it call append_c to add either 0 or 1
        if it's list append the same character to the codes of the list.
      """
      for ele in self.lst:
          ele.append_c(c)

  def append_c(self, c=""):
      """
        Append character to the last two elemnets
      """
      last1 = self.lst[-1]
      last2 = self.lst[-2]
      if isinstance(last1, Element):
          last1.append_c("0")
      if isinstance(last1, ListElements):
          last1.append_all("0")

      if isinstance(last2, Element):
          last2.append_c("1")
      if isinstance(last2, ListElements):
          last2.append_all("1")

  def _unpack(self):
      """
      put all of them in just one list, and return the new one
      """
      unpacked_list = []
      for ele in self.lst:
          if isinstance(ele, Element):
              unpacked_list.append(ele)
          elif isinstance(ele, ListElements):
              unpacked_list.extend(ele.unpack())
      return unpacked_list

  def unpack(self):
      """
        Helper function to _unpack
      """
      return ListElements(self._unpack())

  def calc_avg_length(self):
      """
      return avg length of codes
      """
      avg = sum([x.prob * len(x.code) for x in self.lst])
      return avg

  @classmethod
  def get_list_of_elements(cls, probs, lets, is_100=True):
      """
      static function that ease the way of creating an object
      """
      lst_ele = []
      for p, t in zip(probs, lets):
          p = p / 100 if is_100 == True else p
          element = Element(p, t)
          lst_ele.append(element)
      return cls(lst_ele)

  def fine_print(self):
      """
        print the result in good way
      """
      for ele in self.lst:
          print(ele)

  def get_fine_dict(self):
      """
      return a dict of element for easy access
      """
      from collections import defaultdict
      d = defaultdict()
      for ele in self.lst:
          d[ele.let] = ele.code
      return d

  def get_probs_results(self):
      """
      return probs and results as just an array for further calculations
      """
      probs = []
      results = []
      for ele in self.lst:
          probs.append(ele.prob)
          results.append(ele.code)
      return probs, results

  def print_stats(self):
      """
        prrint the results of "calc_statistics"
      """
      infos, infos_diffs, entropy, avg_len, eff = self.calc_statistics(*self.get_probs_results())
      print("information for each elemnt: {}".format(infos))
      print("Entropy= {:.4f} \nAverage length= {:.4f} \nEfficiency= {:.4f}%".format(entropy, avg_len, eff))

  def calc_statistics(self, probs, result):
      """
      Calculates information in each elemnt, entropy, and avg length, and difference between informatoin adn length of code for each element
      """
      import math
      import numpy as np
      infos = [math.log2(1 / x) for x in probs]
      infos = [f'{item:.3f}' for item in infos]

      entropy = -np.sum(np.multiply(probs, np.log2(probs)))
      avg_len = np.sum(np.multiply(probs, np.array([len(x) for x in result])))
      eff = (entropy / avg_len) * 100
      infos_diffs = [len(x) - float(y) for (x, y) in zip(result, infos)]
      infos_diffs = [f'{item:.3f}' for item in infos_diffs]
      return infos, infos_diffs, entropy, avg_len, eff

  def print_summary(self):
      self.fine_print()
      self.print_stats()




def run_huffman_algoirthm_(lst_probs):
    if len(lst_probs) == 1: return  # Break the recursion
    lst_probs.sort_me()
    lst_probs.append_c()
    lst_probs.trunc_last()
    run_huffman_algoirthm_(lst_probs)  # I'm assuming that I work with list which is mutable


def split_code_chars_probs(comp_code):
    all_keys = Counter(comp_code).keys()  # equals to list(set(words))
    all_probs =  np.array(list(Counter(comp_code).values()))   # counts the elements' frequency
    all_probs = all_probs /  len(all_probs)
    return all_keys, all_probs

def run_huffman_algoirthm(code):
    lets, probs = split_code_chars_probs(code)
    lst = ListElements.get_list_of_elements(probs, lets, is_100=False)
    run_huffman_algoirthm_(lst)
    new_lst = lst.unpack()
    new_lst.sort_me()
    print(new_lst)
    new_lst.fine_print()
    # print("Average length", new_lst.calc_avg_length())
    d = new_lst.get_fine_dict()
    # print("dict", d)
    return d

def encode_huffman(data):
    codes = run_huffman_algoirthm(data)
    op = []
    for c in data:
        c_code = codes[c]
        op.append(c_code)
    output_string = "".join(op)
    return codes ,output_string


def decode_huffman(coded_data, d):
    i = 0
    all_prev_code = ""
    output = []
    while True:
        current_bit = coded_data[i]
        all_prev_code = all_prev_code + current_bit
        if all_prev_code in d.keys():
            output.append(d[all_prev_code])
            all_prev_code = ""
        i +=1
    return output



