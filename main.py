from transformers import BertTokenizerFast,RobertaTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from math import ceil
import time
from utils import *
from model.model import CustomBERTModel,CustomRoBERTaModel
from load_fr_dataset import loader

def train(model,optimizer,lr_scheduler,test_loader,data_loader,epochs,batch_valid):
  model.train()
  best_loss = 1000.0
  #model.zero_grad() 
  
  for epoch in epochs:
    t0 = time.time()
    print("\nEpoch: " + str(epoch+1) + "\n")
    counter = 0
    count_loss = 0
    for batch in chunks(data_loader,8):
        counter += 1
        
        t1 = time.time()
        printProgressBar(counter, nb_chunks, prefix = 'Progress:', suffix = '', length = 50,t_done = t1-t0)

        if counter%batch_valid==0:
            best_loss = validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid,best_loss)
            print("Learning rate: " + str(optimizer.param_groups[0]["lr"])+"\n")
            count_loss = 0

        data = [sent for sent,_ in batch]
        targets = torch.tensor([id for sent,id in batch])
        encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=512)

        model.train()
        optimizer.zero_grad() 
        

        loss, _ = model(**encoding,labels = targets)

        # Prevent gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(),1.0,)
        


        loss.backward()
        count_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()



if __name__ == "__main__":

  
  """# Data loading"""
  # data_loader = get_hatexplain_data('train')
  # test_loader = get_hatexplain_data('test')

  data_loader,test_loader = loader()

  nb_chunks = ceil(len(data_loader)/8)

  """# Tokenizer"""
  model_name = "bert-base-uncased"
  model_name = "camembert-base"
  tokenizer = RobertaTokenizerFast.from_pretrained(model_name, do_lower_case=True)

  """# Train"""
  epochs = [0,1,2]
  model = CustomRoBERTaModel(bert_name  = model_name) 
  #device = torch.device("cpu")
  #model.to(device) ## if gpu


  optimizer = get_optimizer(model)

  # Use scheduler from transformers library
  lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500, num_training_steps=nb_chunks*len(epochs),)

  # Begin the training
  train(model,optimizer,lr_scheduler,test_loader,data_loader,epochs,40)