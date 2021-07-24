from datasets import load_dataset
import numpy as np
from transformers import BertTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from math import ceil
import time
from utils import *
from model.model import CustomBERTModel




def train(model,optimizer,lr_scheduler,epochs,batch_valid):
  model.train()
  #model.zero_grad() 
  t0 = time.time()
  for epoch in epochs:
    print("\nEpoch: " + str(epoch) + "\n")
    counter = 0
    count_loss = 0
    for batch in chunks(data_loader,8):
        counter += 1

        t1 = time.time()
        printProgressBar(counter, nb_chunks, prefix = 'Progress:', suffix = 'Complete', length = 50,t_done = t1-t0)

        if counter%batch_valid==0:
            validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid)
            count_loss = 0



        data = [sent for sent,id in batch]
        targets = torch.tensor([id for sent,id in batch])
        encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=512)

        model.train()
        optimizer.zero_grad() 
        

        loss, outputs = model(**encoding,labels = targets)
        nn.utils.clip_grad_norm_(model.parameters(),1.0,)
        outputs = F.softmax(outputs, dim=1)


        loss.backward()
        count_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()









if __name__ == "__main__":

  dataset_test = load_dataset('hatexplain', split='test')
  valid_texts, valid_labels = convert_dataset_to_list(dataset_test)
  test_loader = [(valid_texts[i],valid_labels[i]) for i in range(len(valid_texts))]
  """# Data loading"""
  dataset_train = load_dataset('hatexplain', split='train')
  train_texts, train_labels = convert_dataset_to_list(dataset_train)
  data_loader = [(train_texts[i],train_labels[i]) for i in range(len(train_texts))]

  """# Tokenization"""

  model_name = "bert-base-uncased"
  tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


  """# Train"""
  epochs = [0,1,2]
  model = CustomBERTModel() # You can pass the parameters if required to have more flexible model
  
  #device = torch.device("cpu")
  #model.to(device) ## if gpu


  decay_parameters = get_parameter_names(model, [nn.LayerNorm])
  decay_parameters = [name for name in decay_parameters if "bias" not in name]
  # self.args.weight_decay

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
      "lr": 5e-05,
  }

  # torch.optim.lr_scheduler.LambdaLR()



  # print(optimizer_grouped_parameters[0])

  # print(optimizer)

  #model.to(device)

  nb_chunks = ceil(len(data_loader)/8)

  optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

  lr_scheduler = get_scheduler(
                  "linear",
                  optimizer,
                  num_warmup_steps=500,
                  num_training_steps=nb_chunks*len(epochs),
              )

  # Begin the training
  train(model,optimizer,lr_scheduler,epochs,2)