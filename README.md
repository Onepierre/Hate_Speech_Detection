Train a Bert model for hate speech detection
Based on the dataset ```hatexplain``` (https://arxiv.org/abs/2012.10289)

# Model
Fine tunes a Bert-based-uncased model to label sentences into 3 groups :   
```No problem```  
```Offensive language```  
```Hate speech```  

Model is composed of a 12-Layers Bert model, then a Linar Layer is put after the `[CLS]` output
(https://arxiv.org/pdf/2012.10289v1.pdf)
# Setup
```python -m pip install -r requirements.txt```

# Train
```python main.py```  
You can also use the transformers trainer with :
```python main_transformers_trainer.py``` (Not tested for now, it may contain bugs)