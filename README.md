# DeepActiveCRISPR

## Introduction

DeepActiveCRISPR is a deep learning based prediction model for sgRNA on-target efficacy prediction. This model is based on the carefully designed convolutional neural network for model training and prediction. In order to achieve higher AUC performance on only small amount of training data, DeepActiveCRISPR adopts an active learning and transfer learning based paradigm for model building and sgRNA on-target efficacy prediction. We derived several pre-trained models on existed cell line data, while user can fine-tune the neural network based on these pre-trained models for the prediction of sgRNA efficacy on new cell lines. Also, with the help of active learning technique, DeepActiveCRISPR can automatically select the most model-benefical un-labeled sgRNAs from the pool for the experiment experts to label. These sgRNAs, if their efficacy were labeled first, can contribute most to the model training. In summary, DeepActiveCRISPR provides a general computational framework for sgRNA on-target efficacy prediction based on limited training data for new cell types. Also it provides an active learning based schema for the experiment experts to select the sgRNAs to be labeled. This paradigm provides the first prototype study for the optimized sgRNA on-target design with saved training data and saved experimental label cost.

## Requirement

* python == 3.6
* tensorflow == 1.3.0 
* sonnet == 1.9

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

### 1. Transfer Learning

#### 1.1 ontar_raw_cnn_finetune_freeze.py

　　* Train your own CNN model based on our pre-trained model.

　　* Freeze several layers of the neural network while training (you can change the specific settings manually).

　　* Do not forget to change the file path before executing.

#### 1.2 ontar_raw_cnn_finetune.py

　　* Train your own CNN model based on our pre-trained model.

　　* Will fine-tune all layers of the neural network.

　　* Do not forget to change the file path before executing.

#### 1.3 ontar_raw_cnn_pretrain.py

　　* Train and save your own pre-trained model.

#### 1.4 ontar_raw_cnn_transfer.py

　　* Test the performance of plain transfer training (Training on dataset A while testing on dataset B).

### 2. Active Learning

#### 2.1 active_cnn.py

　　* Train your own CNN model based on our pre-trained model.

　　* Using active learning technique to select the most valuable training data during each fine-tune process.

　　* You can change the parameters of active learning algorithm manually.

#### 2.2 randomselect_cnn.py

　　* Train your own CNN model based on our pre-trained model.

　　* Randomly select training data during each fine-tune process in order to compare the performance.

### 3. Data type

```
CGGTAGAAGCAGGTAGTCTGGGG	AAAANNNNNNNNNNNNNNNNNNN	AAAAAAAAAAAAAAAAAAAAAAA	AAAAAAAAAAAAAAAAAAAAAAA	NNNNNNNNNNNNNNNNNNNNNNN	1
DNA sequence --- Epigenetic Sequence(not used yet) --- efficacy(0 or 1)  
```
