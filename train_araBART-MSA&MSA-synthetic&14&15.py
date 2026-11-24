# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:08:32 2024

@author: Karim
"""

import pandas as pd
# read the 2 datasets and combine them

#MSA_data=pd.read_csv('D:/Phd/Part 2-1/other data/MSA_data_arat5_seq2seq.csv')
MSA_data_qalb_train_14_15=pd.read_csv('D:/Phd/Optimization/MSA&qalb_14_15_training_g.csv')
MSA_Synthetic=pd.read_csv('D:/Phd/Optimization/sythetic_(msa)aragpt2large(arat5)_Qlora4(seq2seq)_g.csv')

training_data=pd.concat([MSA_data_qalb_train_14_15,MSA_Synthetic],axis=0)
training_data.to_csv('MSA&MSA_synthetic&(gpt2_large_Qlora4)(arat5)qalb_14_15_training_g.csv',index=False,encoding='utf-8-sig')

from happytransformer import HappyTextToText
happy_tt = HappyTextToText("ARABART", "moussaKam/AraBART")

from happytransformer import TTTrainArgs, TTEvalArgs

args = TTTrainArgs(batch_size=1, num_train_epochs=20 ,max_input_length=2000,max_output_length=2000)
happy_tt.train('MSA&MSA_synthetic&(gpt2_large_lora4)qalb_14_15_training.csv', args=args)

happy_tt.save("trained_2000_30_QALB2014&QALB2015_arabart-11")