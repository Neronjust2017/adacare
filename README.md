# AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration

The source code for *AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration*

Thanks for your interest in our work. More details about the experiment, case study and visualization system will be released before AAAI2020.

## Visualization
Welcome to test the prototype of our visualization tool:

http://47.93.42.104/215 (Cause of death: CVD)  
http://47.93.42.104/318 (Cause of death: GI disease)  
http://47.93.42.104/616 (Cause of death: Other)  
http://47.93.42.104/265 (Cause of death: GI disease)  
http://47.93.42.104/812 (Cause of death: Cachexia)  
http://47.93.42.104/455 (Cause of death: CVD)  
http://47.93.42.104/998 (Alive)  
http://47.93.42.104/544 (Alive)  

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Data preparation
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run decompensation prediction task on MIMIC-III bechmark dataset, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the decompensation dataset, please save the files in ```decompensation``` directory to ```data/``` directory.

## Fast way to test AdaCare with MIMIC-III
1. We provide the trained weights in ```./saved_weights/AdaCare```. You can obtain the reported performance in our paper by simply load the weights to the model.

2. You need to run ```train.py``` in test mode and input the data directory. For example,

    ```$ python train.py --test_mode=1 --data_path='./data/' ```

## Training AdaCare
1. The minimum input you need to train AdaCare is the dataset directory and file name to save model. For example,

    ```$ python train.py --data_path='./data/' --file_name='trained_model' ```

3. You can also specify batch size ```--batch_size <integer> ```, learning rate ```--lr <float> ``` and epochs ```--epochs <integer> ```

4. Additional hyper-parameters can be specified such as the dimension of RNN, dropout rate, etc. Detailed information can be accessed by 

    ```$ python train.py --help```

5. When training is complete, it will output the performance of AdaCare on test dataset.
