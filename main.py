from logging import raiseExceptions
from transformers import BertTokenizerFast,RobertaTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from math import ceil
import time
from utils import *
from model.model import CustomBERTModel,CustomRoBERTaModel,CustomCamemBERTModel,MultiLabelCamemBERTModel
from load_fr_dataset import loader,loader_multi_label

def train(model,tokenizer,optimizer,lr_scheduler,test_loader,data_loader,epochs,batch_valid, resume_training = None):
  best_loss = 1000.0
  if not resume_training == None: 
    new_epoch = resume_training["epoch"]
    new_turn = resume_training["batch"]
    counter = 0
    count_loss = 0
    best_loss = validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid,best_loss)
  model.train()
  nb_chunks = ceil(len(data_loader)/8)
  
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
      if not resume_training == None:
        if new_epoch > epoch or (new_epoch == epoch and new_turn/8 >= counter):
          optimizer.step()
          lr_scheduler.step()
          continue
      

      if counter%batch_valid==0:
          best_loss = validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid,best_loss)
          print("Learning rate: " + str(optimizer.param_groups[0]["lr"])+"\n")
          count_loss = 0

      data = []
      for sent,_ in batch:
        if len(sent) > 300:
          data.append(sent[:300])
        else:
          data.append(sent)
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
  validation_score(model,tokenizer,test_loader,data_loader,counter,count_loss,batch_valid,best_loss)
  print("Training ended")


def do_training(checkpoint = None, epoch = -1, batch = -2):
  if checkpoint == None or epoch == -1 or batch == -2:
    print("Starting training from the beginning. If this is not wanted, you may have forgotten arguments")
    checkpoint = None
    epoch = 0
    batch = -1
  else:
    print("Resume training from epoch " + str(epoch)+" and input " + str(batch))
    epoch = epoch -1

  resume_training = {"epoch":epoch,"batch":batch}

  """# Data loading"""
  language = "en"
  
  if language == "en":
    data_loader = get_hatexplain_data('train')
    test_loader = get_hatexplain_data('test')
    model_name = "bert-base-uncased"
    model_name = "roberta-base"

  if language == "fr":
    #data_loader,test_loader = loader()
    data_loader,test_loader =loader_multi_label()
    model_name = "camembert-base"

  nb_chunks = ceil(len(data_loader)/8)

  #tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
  tokenizer = RobertaTokenizerFast.from_pretrained(model_name, do_lower_case=True)

  #model = CustomBERTModel(bert_name  = model_name) 
  model = CustomRoBERTaModel(bert_name  = model_name) 
  # model = CustomCamemBERTModel() 
  #model = MultiLabelCamemBERTModel() 
  if not checkpoint == None:
    model.load_state_dict(torch.load(checkpoint))
  #device = torch.device("cpu")
  #model.to(device) ## if gpu

  epochs = [0,1,2]

  # Load optimizer
  optimizer = get_optimizer(model,5e-05)

  # Use scheduler from transformers library
  lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500, num_training_steps=nb_chunks*len(epochs),)

  # Do the training
  train(model,tokenizer,optimizer,lr_scheduler,test_loader,data_loader,epochs,100,resume_training)
  



if __name__ == "__main__":
  # TRAINING EXAMPLE
  # checkpoint = "model_save/model.ckpt"
  # epoch = 2
  # batch = 13600
  # do_training(checkpoint, epoch, batch)
  do_training()