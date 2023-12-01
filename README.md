# TechMova
Sentiment Analysis with Deep Learning using BERT

import torch
import pandas as pd
from tqdm.notebook import tqdm
import warnings
 
warnings.filterwarnings("ignore", category=FutureWarning) 

df= pd.read_csv(
    "/Users/elenarepetskaya/NLP sentiment analysis BERT/smileannotationsfinal.csv",
    names =["id", "text", "category"]
)
#set unique index to Id
df.set_index("id",inplace=True)

#print(df.head())
#print(df.text.iloc[0])
# print(df.category.value_counts())

#sacar nocode (emocion no determinada) y multi emociones con separador sad|disgust
df = df[~df.category.str.contains("\|")]
df = df[df.category != "nocode"]
print(df.category.value_counts())

#marcamos label_dict para emociones de 0 a 5 en este caso y 
# lo ponemos en una columna extra y continuacion le cambiamos el nombre a label
possible_labels = df.category.unique()
label_dict= {}
for index, possible_label in enumerate (possible_labels):
    label_dict[possible_label]= index
print(label_dict)

df["label"]= df.category.replace(label_dict)
#print(df.head(10))

#Training/Validation Split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.15,
    random_state=17,
    stratify=df.label.values #seleccione de cada caracteristica por igual
)
#nueva columna data_type
df["data_type"]=["not_set"]* df.shape[0]
print(df.head(10))

#statify para que saque x_train y X_val por igual de cada caracterisica 
df.loc[X_train, "data_type"] = "train"
df.loc[X_val, "data_type"] = "val"

#agrupamos y contamos
df.groupby(["category", "label", "data_type"]).count()
#print(df.head(30))

#LOADING TOKENIZER AND ENCODING OUR DATA

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=="train"].text.values,
    add_special_tokens=True,
    #bert indica donde comienza y donde acaba sentence(frase)
    return_attention_mask=True, #para igualar la cantidad de las palabras en una frase
    #igualar la dimencion de todas frases
    pad_to_max_length=True,
    max_length=256, #longitud maxima de palabras
    return_tensors="pt" #pytorch
)
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=="val"].text.values,
    add_special_tokens=True,
    #bert indica donde comienza y donde acaba sentence(frase)
    return_attention_mask=True, #para igualar la cantidad de las palabras en una frase
    #igualar la dimencion de todas frases
    pad_to_max_length=True,
    max_length=256, #longitud maxima de palabras
    return_tensors="pt" #pytorch
)
input_ids_train = encoded_data_train["input_ids"]#accedemos al diccionario 
attention_masks_train = encoded_data_train["attention_mask"]
labels_train = torch.tensor(df[df.data_type =="train"].label.values)

input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
labels_val = torch.tensor(df[df.data_type=="val"].label.values)

dataset_train = TensorDataset(input_ids_train, 
                                attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, 
                                attention_masks_val, labels_val)

len(dataset_train)
#print(len(dataset_train))
len(dataset_val)
#print(len(dataset_val))

#SETTING UP BERT PRETRAIN MODEL
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
     num_labels = len(label_dict),
     output_attentions=False,
     output_hidden_states=False
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size=4 #32
dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)
dataloader_val = DataLoader(
    dataset_val,
    sampler=SequentialSampler(dataset_val),
    batch_size=32
)

#SETTING UP OPTIMIZER AND SCHEDULE
from transformers  import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(
    model.parameters(),
    lr=1e-5, #2e-5>5e-5
    eps=1e-8 #epsilon
)
epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train)*epochs

)
#DEFINING OUR PERFORMANCE METRICS

import numpy as np
from sklearn.metrics import f1_score

#preds [1 0 0 0 0 0] predicir la probilidad
#check tutorial 

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")#para tener en cuenta que los clases estan desbalanceadas

def accurancy_per_class (preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}# key value debe ser separado por :

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]#numpy build indexing. solo escogemos preds_flat donde labels_flat es igual a label
        y_true = labels_flat[labels_flat==label]
        print(f"Class: {label_dict_inverse[label]}")
        print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = { 
                    "input_ids":       batch[0],
                    "attention_mask":   batch[1],
                    "labels":           batch[2] #revisar coma
                }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]#use a logits as prediccion 
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):
    
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train,
                        desc="Epoch {:1d}".format(epoch),
                        leave=False,
                        disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids"      : batch[0],
            "attention_mask" : batch[1],
            "labels"         : batch[2]
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"training_loss" : "{:.3f}".format(loss.item()/len(batch))})

    torch.save(model.state_dict(), f"Models/Bert_ft_epoch{epoch}.model")
    
    tqdm.write("\nEpoch {epoch}")

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 score (weighted): {val_f1}")


#TRAINING Y EVALUATING OUR MODEL
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                            num_labels=len(label_dict),
                                                            output_attentions=False,
                                                            output_hidden_states=False)
model.to(device)
pass

model.load_state_dict(
    torch.load("ruta al modelo",
    map_location=torch.device("cpu"))
)
_, predictions, true_vals = evaluate(dataloader_val)

accuracy_per_class(predictions, true_vals)










