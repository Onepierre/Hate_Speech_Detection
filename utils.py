
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

def convert_dataset_to_list(df):
  sentences_list = []
  labels_list = []
  for i in df:
    text = ""
    for word in i['post_tokens']:
      text += " "
      text+= word
    sentences_list.append(text)  
    labels = i['annotators']["label"]
    value = np.argmax(np.bincount(labels))
    labels_list.append(value)
  return sentences_list,labels_list

def validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid):


  print(str( 8*counter) + "/" + str(len(data_loader)))

  model.eval()
  count = 0
  juste = 0
  out = []
  targets = []
  loss_sum = []
  
  nb_chunks_val = ceil(len(test_loader)/20)
  chunk = chunks(test_loader,20)

  print("Test val")
  printProgressBar(0, nb_chunks_val, prefix = 'Progress:', suffix = 'Complete', length = 50)
  count = 0
  t0_test = time.time()
  for batch in chunk:
    target_temp = []
    for i in [id for _,id in batch]:
      target_temp.append(i)
      targets.append(i)
    count += 1 
    t1_test = time.time()
    t_done_test = t1_test-t0_test
    printProgressBar(count, nb_chunks_val, prefix = 'Progress:', suffix = 'Complete', length = 50, t_done=t_done_test)  
    data = [sent for sent,id in batch]

    with torch.no_grad():  
      encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,max_length=512)
      loss, outputs = model(**encoding,labels = torch.tensor(target_temp))
      outputs = F.softmax(outputs, dim=1)
      loss_sum.append(loss.item())
    for i in outputs:
      out.append(i.numpy())

  for i,a in enumerate(out):
    if a.argmax() == targets[i]:
      juste += 1
  print("Accuracy: " + str(juste/len(out)))
  print("Test Loss: " + str(np.mean(np.array(loss_sum))))
  print("Train Loss: " + str(count_loss/batch_valid))



