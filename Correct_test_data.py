# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:01:08 2024

@author: Karim
"""

from happytransformer import HappyTextToText
import pandas as pd

happy_tt = HappyTextToText("ARAT5", "D:/Phd/Part 2-1/other data/AraT5_MSA&sysnthetic(medium)&QALB14&15")

Test_data=pd.read_csv('D:/Phd/Part 2-1/other data/Arabic_l2_test-g.csv')

from happytransformer import TTSettings

#beam_settings =  TTSettings(num_beams=7,max_length=2400,min_length=5)
top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.9,max_length=2400,min_length=5)
Test_generated= pd.DataFrame()

for index, row in Test_data.iterrows():
    prediction= happy_tt.generate_text(row['input'], args=top_p_sampling_settings)  
    print(prediction.text)
    pred = pd.Series(prediction.text)
    #Test_generated=Test_generated.append(pred, ignore_index=True)
    Test_generated=pd.concat([Test_generated,pred],axis=0)
   
   
Test_generated.to_csv("generated_l2_msa&systhetic(medium)_arat5_beam.csv", header=False, index=False)    