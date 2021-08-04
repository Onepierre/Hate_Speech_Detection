
from datasets import load_dataset
import numpy as np
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

def get_hatexplain_data(name):
  # name = 'test' or 'train'
  dataset_test = load_dataset('hatexplain', split=name)
  valid_texts, valid_labels = convert_dataset_to_list(dataset_test)
  return [(valid_texts[i],valid_labels[i]) for i in range(len(valid_texts))]

def validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid,best_loss):


  print(str( 8*counter) + "/" + str(len(data_loader)))

  model.eval()
  count = 0
  juste = 0
  out = []
  targets = []
  loss_sum = []
  
  nb_chunks_val = ceil(len(test_loader)/20)
  chunk = chunks(test_loader,20)

  print("\nScore on test dataset")
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
    printProgressBar(count, nb_chunks_val, prefix = 'Progress:', suffix = '', length = 50, t_done=t_done_test)  
    data = [sent for sent,id in batch]

    with torch.no_grad():  
      encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,max_length=512)
      loss, outputs = model(**encoding,labels = torch.tensor(target_temp))
      loss_sum.append(loss.item())
    for i in outputs:
      out.append(torch.sigmoid(i).numpy())

  # For single label classification
  for i,a in enumerate(out):
    if a.argmax() == targets[i]:
      juste += 1
  print("Accuracy: " + str(juste/len(out)))

  # For multi label clasification
  # length = len(out[0])
  # juste = [0] * length
  # for i,a in enumerate(out):
  #   for j,b in enumerate(a):
  #     res = 0
  #     if b > 0.5:
  #       res = 1
  #     if res == targets[i][j]:
  #       juste[j] += 1/len(out)
  # print("Accuracy: " + str(juste))


  print("Test Loss: " + str(np.mean(np.array(loss_sum))))
  print("Train Loss: " + str(count_loss/batch_valid))
  if best_loss > np.mean(np.array(loss_sum)):
    best_loss = np.mean(np.array(loss_sum))
    torch.save(model.state_dict(), "model_save/model.ckpt")
    print("Model saved")
  return best_loss

def get_optimizer(model,lr = 5e-05):
  # Retur adamW optimizer for bert custom model
  # I used AdamW from transformers
  decay_parameters = get_parameter_names(model, [nn.LayerNorm])
  decay_parameters = [name for name in decay_parameters if "bias" not in name]

  optimizer_grouped_parameters = [
      {
          "params": [p for n, p in model.named_parameters() if n in decay_parameters],
          "weight_decay": 0.1,
      },
      {
          "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
          "weight_decay": 0.0,
      },
  ]

  optimizer_kwargs = {
      "betas":  (0.9, 0.999),
      "eps": 1e-8,
      "lr": lr,
  }
  

  return AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

