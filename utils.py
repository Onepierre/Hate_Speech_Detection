
from datasets import load_dataset
import numpy as np
from transformers import BertTokenizer,BertTokenizerFast
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from math import ceil
import sys
import time
import datetime

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r",t_done = None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    if not t_done == None:
      t_done = int(t_done)
      t_to_do = int(t_done*(total/iteration - 1))
      time_str = str(datetime.timedelta(seconds=t_done)) + " < " +str(datetime.timedelta(seconds=t_to_do))
    else:
      time_str = ""

    print(f'\r{prefix} |{bar}| {percent}% {time_str} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

