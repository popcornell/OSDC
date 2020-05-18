# OSDC

Overlapped Speech Detection and Counting
---
Installation:
- pip install -r requirements.txt
- python setup.py install 

---
This code has a Kaldi.like structure. 

    egs --> AMI ----> bash scripts 
             |------> local
             |------> conf


In each egs we provide bash scripts for data preparation, training and inferencing.
The scripts are located in local. 
A yaml configuration file in conf is used. 

 