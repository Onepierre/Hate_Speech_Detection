import numpy as np
import pandas as pd
from sklearn import datasets
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import random
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import pickle
from sklearn.metrics import f1_score,roc_auc_score

target_names = ["hate_speech", "Offensive language", "Neither"]

cuda_available = False

def get_prediction(text,model,tokenizer):
    max_length = 512
    # prepare our text into tokenized sequence
    if cuda_available:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    else:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs,probs.argmax()

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)
 
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

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def train():

    #set_seed(1)
    model_name = "bert-base-uncased"
    max_length = 512
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    
    # split dataset

    print("Loading Datasets")

    dataset_train = load_dataset('hatexplain', split='train')
    train_texts, train_labels = convert_dataset_to_list(dataset_train)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    train_dataset = NewsGroupsDataset(train_encodings, train_labels)

    dataset_test = load_dataset('hatexplain', split='test')
    valid_texts, valid_labels = convert_dataset_to_list(dataset_test)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
    valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

    # convert our tokenized data into a torch Dataset
    
    print("Loading model")

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        logging_steps=200,               # log & save weights each logging_steps
        evaluation_strategy="steps",     # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    print("Begin Training")
    # train the model
    trainer.train()

    model_path = "hatexplain-bert-base-uncased-test"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    # evaluate the current model after training
    trainer.evaluate()


def test():
    model_path = "hatexplain-bert-base-uncased"
    if cuda_available:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3).to("cuda")
    else:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)


    # Example #1
    # text = """I hate islam"""
    # probs,pred = get_prediction(text,model,tokenizer)
    # print(target_names[pred])

    # Example #2
    dataset_test = load_dataset('hatexplain', split='test')
    valid_texts, _ = convert_dataset_to_list(dataset_test)
    count = 0
    label_predict = []
    score_predict = []

    for i,text in enumerate(valid_texts):
        probs, pred = get_prediction(text,model,tokenizer)
        label_predict.append(pred)
        score_predict.append(probs.detach().numpy()[0])

    with open("y_pred.txt", "wb") as fp:   #Pickling
        pickle.dump(label_predict, fp)

    with open("y_score.txt", "wb") as fp:   #Pickling
        pickle.dump(score_predict, fp)
 
def scores():
    dataset_test = load_dataset('hatexplain', split='test')
    _, valid_labels = convert_dataset_to_list(dataset_test)

    with open("y_pred.txt", "rb") as fp:   # Unpickling
        y_pred = pickle.load(fp)

    with open("y_score.txt", "rb") as fp:   # Unpickling
        y_score = pickle.load(fp)


    # for i,pred in enumerate(y_pred):
    #     print(target_names[pred])
    #     print(target_names[valid_labels[i]])


    f1 = f1_score(valid_labels[0:len(y_pred)], y_pred, average='macro')
    print("f1_score : "+str(f1))

    auroc = roc_auc_score(valid_labels[0:len(y_score)], y_score, multi_class='ovr')
    print("auroc : "+str(auroc))

scores()