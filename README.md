# DeepActiveCRISPR

## Introduction

DeepActiveCRISPR is a deep learning based prediction model for sgRNA on-target efficacy prediction. This model is based on convolutional neural network for model training and prediction. In order to achieve higher AUC performance on only small amount of training data, we adopted active learning and transfer learning paradigm. We derived several pre-trained model on different cells, while user can fine-tune the neural network based on these pre-trained model. Also, with the help of active learning technique, the program can automatically select the most useful training data from the pool, which will contribute the most to the model in the training process. 

## Requirement

* python == 
* tensorflow == 
* sonnet == 

## Directory Structure:

```
.
|-cnn
  |-active
    |  active learning CNN classifier
    |- active_cnn.py 							
    |- randomselect_cnn.py 						
  |-classifier
    |  original CNN classifier
    |- ontar_raw_cnn.py 						
    |- ontar_raw_cnn_epi.py 					
    |- ontar_raw_cnn_mixed_t.py 				
  |-transfer
    |  fine-tune CNN classifier
    |- ontar_raw_cnn_finetune.py 				
    |- ontar_raw_cnn_finetune_freeze.py 		
    |- ontar_raw_cnn_pretrain.py 				
    |- ontar_raw_cnn_transfer.py 				
  |-premodel
    |  pre-trained model
|-gradient
  |-active
    |  active learning on GradientBoost classifier
  |-transfer
    |  transfer learning on GradientBoost classifier
```

## Usage


