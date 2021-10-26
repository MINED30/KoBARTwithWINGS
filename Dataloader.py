
import re
import torch 
import torch.nn as nn

class GapSentenceGeneration(torch.utils.data.Dataset):
  def __init__(self,verbose=False):
    self.tokenizer = tokenizer
    self.df = df
    self.verbose = verbose
  
  def __getitem__(self,idx):
    doc = df.iloc[idx]['context']
    docs = '. ;'.join(doc.split('. '))
    docs = '? ;'.join(docs.split('? '))
    docs = [s.strip() for s in docs.split(';') if s != ""]

    selected_idx = [int(id) for id in df.iloc[idx]['rouge_selected'][1:-1].split(',')]
    target_text_list = []
    for si in selected_idx:
      target_text_list.append(docs.pop(si))
      docs.insert(si,"<GSG>")

    encoder_input_text = re.sub("\ *<GSG>\ *",'<GSG>',' '.join(docs))
    decoder_input_text = re.sub("\ *<s>\ *",'<s>',' '.join(["<s>"+s for s in target_text_list]))
    decoder_target_text = re.sub("\ *</s>\ *",'</s>',' '.join([s+"</s>" for s in target_text_list]))
    if self.verbose:
      print("encoder_input_text :",encoder_input_text)
      print("decoder_input_text :",decoder_input_text)
      print("decoder_target_text :",decoder_target_text)

    encoder_input_text = tokenizer.batch_encode_plus([encoder_input_text],padding='max_length', max_length=512, return_tensors="pt")
    decoder_input_text = tokenizer.batch_encode_plus([decoder_input_text],padding='max_length', max_length=512, return_tensors="pt")
    decoder_target_text = tokenizer.batch_encode_plus([decoder_target_text],padding='max_length', max_length=512, return_tensors="pt")
    return {"encoder_input_text": encoder_input_text, 
            "decoder_input_text": decoder_input_text, 
            "decoder_target_text": decoder_target_text}

  def __len__(self):
      return self.df.__len__()

  def __repr__(self):
    return f"data size : {self.__len__()} "
