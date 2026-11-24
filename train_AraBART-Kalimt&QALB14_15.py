# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:53:55 2024

@author: Karim
"""
import pandas as pd
# read the 2 datasets and combine them

kalimat_data=pd.read_csv('D:/Phd/Part 2-1/kalimat_data_seq2seq.csv')
QALB_data=pd.read_csv('D:/Phd/Part 2-1/QALB_2014&2015_train.csv')

training_data=pd.concat([QALB_data,kalimat_data],axis=0)
training_data.to_csv('kalmiat&qalb_14_15_training.csv',header=None,index=False)

from happytransformer import HappyTextToText
happy_tt = HappyTextToText("ARABART", "moussaKam/AraBART")

from happytransformer import TTTrainArgs, TTEvalArgs

args = TTTrainArgs(batch_size=1, num_train_epochs=20 ,max_input_length=2000,max_output_length=2000)
happy_tt.train('D:/Phd/Part 2-1/kalmiat&qalb_14_15_training.csv', args=args)

happy_tt.save("trained_2000_30_QALB2014&QALB2015_arabart-11")