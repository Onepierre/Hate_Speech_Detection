from logging import raiseExceptions
from transformers import BertTokenizerFast,RobertaTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from utils import *
from model.model import CustomBERTModel,CustomRoBERTaModel,CustomCamemBERTModel
from load_fr_dataset import loader



def evaluate(test_loader):
    # Evaluate the accuracy on a loader of data
    # test_loader is a list of tuples (sentence,label)
    model.eval()
    count = 0
    juste = 0
    out = []
    targets = []
    
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
        printProgressBar(count, nb_chunks_val, prefix = 'Progress:', suffix = '', length = 50, t_done=time.time()-t0_test)  
        data = [sent for sent,id in batch]

        with torch.no_grad():  
            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,max_length=512)
            _, outputs = model(**encoding,labels = torch.tensor(target_temp))
            outputs = F.softmax(outputs, dim=1)
            for i in outputs:
                out.append(i.numpy())

    for i,a in enumerate(out):
        if a.argmax() == targets[i]:
            juste += 1
    print("Accuracy: " + str(juste/len(out)))

def predict(sentences):
    # Predict labels of sentences 
    # sentences is a list of sentences
    model.eval()
    count = 0
    logits = []
    
    nb_chunks_val = ceil(len(sentences)/20)
    chunk = chunks(sentences,20)

    print("\nScore on test dataset")
    count = 0
    t0_test = time.time()
    for batch in chunk:
        count += 1 
        printProgressBar(count, nb_chunks_val, prefix = 'Progress:', suffix = '', length = 50, t_done=time.time()-t0_test)  
        with torch.no_grad():  
            encoding = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True,max_length=512)

            outputs = model(**encoding)[0]
            outputs = F.softmax(outputs, dim=1)
            for i in outputs:
                logits.append(i.numpy())
    return logits

if __name__ == "__main__":
    model_to_test = "camembert"
    labels = ["Normal", "Offensive", "Hateful"]
    if model_to_test == "camembert":
        # TESTING
        model_name = "camembert-base"

        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, do_lower_case=True)
        model = CustomCamemBERTModel() 
        model.load_state_dict(torch.load("model_save/camembert(768-50-3).ckpt"))
        model.eval()

        _,test_loader = loader()
        # evaluate(test_loader)

        sentences = []
        for i,_ in test_loader:
            sentences.append(i)
        logits = predict(sentences)
        for i in range(len(logits)):
            #print(sentences[i])
            id = np.argmax(np.array(logits[i]))
            #print(labels[id])
            if id == 2:
                print("aaaaaaa")