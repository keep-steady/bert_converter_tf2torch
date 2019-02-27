# bert_converter_tf2torch
BERT model, tensorflow2pytorch customize

This is from https://github.com/ttxttx1111/bert.git
You can get a Bert model below
'''
!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip multi_cased_L-12_H-768_A-12.zip
!rm multi_cased_L-12_H-768_A-12.zip
'''

In 'multi_cased_L-12_H-768_A-12' folder, There are 5 file
1) bert_config.json
2) bert_model.ckpt.data-00000-of-00001
3) bert_model.ckpt.index
4) bert_model.ckpt.meta
5) vocab.txt

Then you can convert tf2torch below, your bert model file name after '--bert_model_path'
'''
python convert_tf_checkpoint_to_pytorch_custom.py --bert_model_path multi_cased_L-12_H-768_A-12
'''

Then you can get a pytorch_model.bin

