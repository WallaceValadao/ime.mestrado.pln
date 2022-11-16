import codecs
codecs.register_error('strict', codecs.lookup_error('surrogateescape'))

from transformers import AutoModel, AutoTokenizer
from keras.preprocessing import sequence
from keras.preprocessing import sequence
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import torch
import numpy as np

import models.util.valorPositivo as valorPositivo
import models.util.tratarCorteFrases as tratarFrases

class PreProcessamentoBert():

    def __init__(self, config, path_model, max_len=512, parsePostiveValues=False):
        self.config = config
        self.path_model = path_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_model, do_lower_case=True)
        self.model = AutoModel.from_pretrained(self.path_model)
        self.max_len = max_len
        self.parsePostiveValues = parsePostiveValues
        
        #self.model.eval()


        # Load pre-trained model (weights)
        #self.model = BertModel.from_pretrained('bert-base-uncased')
        #self.model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased')

        #model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def _tokenBert(self, text): 
        text = text.replace(' .', '. [SEP]')
        text = text.replace(' ,', ',')
        text = text.replace('  ', ' ')

        return text
        #return "[CLS] " + text + " [SEP]"

    
    def preprocessing_for_bert(self, data):
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            encoded_sent = self.tokenizer.encode_plus(
                text=self._tokenBert(sent),  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        #input_ids = torch.tensor(input_ids)
        #attention_masks = torch.tensor(attention_masks)

        #result =  input_ids, attention_masks
        result =  np.array(input_ids)

        result = sequence.pad_sequences(result, maxlen=self.max_len)

        return result


    def bert_encode_base(self, texts):
        all_tokens = []
        all_masks = []
        all_segments = []
        
        for textBase in texts:
            for text in tratarFrases.obterCorteFrases(textBase):
                text = self._tokenBert(text)
                tokenized_text = self.tokenizer.tokenize(text)
                
                tokenized_text = tokenized_text[:self.max_len-2]
                pad_len = self.max_len - len(tokenized_text)
                
                tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens += [0] * pad_len
                pad_masks = [1] * len(tokenized_text) + [0] * pad_len
                segment_ids = [0] * self.max_len
                
                all_tokens.append(tokens[0:self.max_len])
                all_masks.append(pad_masks[0:self.max_len])
                all_segments.append(segment_ids[0:self.max_len])
        
        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


    def bert_encode(self, texts):
        all_tokens, all_masks, all_segments  = self.bert_encode_base(texts)

        return sequence.pad_sequences(all_tokens, maxlen=self.max_len)


    def encoder(self, texts):
        all_tokens = []
        
        for text in texts:
            text = self._tokenBert(text)
            tokenized_text = self.tokenizer.tokenize(text)

            tokenized_text = tokenized_text[:self.max_len-2]
                
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

                print ("Number of layers:", len(encoded_layers))
                layer_i = 0
                
                print ("Number of batches:", len(encoded_layers[layer_i]))
                batch_i = 0
                
                print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
                token_i = 0
                
                print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

                token_embeddings = torch.stack(encoded_layers, dim=0)
                token_embeddings.size()

                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings.size()

                token_embeddings = token_embeddings.permute(1,0,2)
                token_embeddings.size()

                # shape [22 x 768]
                token_vecs = encoded_layers[11][0]
                
                # 22 token vectors.
                sentence_embedding = torch.mean(token_vecs, dim=0)

                array_embeddings = sentence_embedding.detach().cpu().numpy()

                array_embeddings = array_embeddings[:self.max_len]

                for i in range(0, len(array_embeddings)):
                    if array_embeddings[i] < 0:
                        array_embeddings[i] *= -1;

                all_tokens.append(array_embeddings)

        return all_tokens


    def encoderBal(self, texts):
        all_tokens = []

        for textBase in texts:
            resultText = []
            for text in tratarFrases.obterCorteFrases(self.config, textBase, self.max_len):
                text = self._tokenBert(text)

                #text = text[:512]
          
                input_ids = self.tokenizer.encode(text, return_tensors='pt')
                
                with torch.no_grad():
                    #outs = model(input_ids)
                    encoded_layers = self.model(input_ids)
                    encoded = encoded_layers[0][0, 1:-1]

                    # shape [22 x 768]
                    #token_vecs = encoded_layers[11][0]
                    
                    # 22 token vectors.
                    sentence_embedding = torch.mean(encoded, dim=0)

                    array_embeddings = sentence_embedding.detach().cpu().numpy()

                    array_embeddings = array_embeddings[:self.max_len]

                    resultText.append(list(array_embeddings))
            all_tokens.append(resultText)

        bertArray = valorPositivo.converterArray(all_tokens)

        return list(bertArray)
    

    def getAttributesBase(self, previsores):
        #result =[]

        #for item in previsores:
        #    result.append(self.teste2(item))

        return self.encoderBal(previsores)
        #return self.bert_encode(previsores)
