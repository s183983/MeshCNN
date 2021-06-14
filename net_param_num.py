# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:46:12 2021

@author: lowes
"""

'''
To do: 
Lav et script, med input [mappe1,mappe2,...]
hvor hvert net fra mapperne loades og antal parametre udregnes
disse skal med i tabellerne for resultater

Udregn tid for et forward pass (på thinlinc) evt. for C5

For C5 kør test med phase validation og test, hvor acc og dice gemmes
så de kan laves til histogrammer.
    
    
'''

def num_of_params(net,full_print=False):
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            n_param += param.data.numel()
            if full_print:
                print(name+", shape="+str(param.data.shape))
    print("Net has " + str(n_param) + " params.")
    return n_param

main_folder = 'C:\\Users\\lowes\\OneDrive\\Skrivebord\\DTU\\6_Semester\\Bachelor\\nets_to_count'
