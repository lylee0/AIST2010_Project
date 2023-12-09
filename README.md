# AIST2010_Project
Predicting Singers from Songs through Machine Learning

Packages required<br />
Please refer to requirements.txt

Firstly Download the dataset in the link below:<br /> 
https://mycuhk-my.sharepoint.com/:f:/g/personal/1155158772_link_cuhk_edu_hk/Egbl1_rh3NtGlnzfTuX5f2kBDVBFU-3HVHaV5hnBEZnyQw?e=u7x6Tl

Directory Set-up<br />
----main_directory<br />
    get_mfcc.py<br />
    y_labels_2500.csv
    model.py<br />
    ----dataset<br />
        1.wav<br />
        2.wav<br />
        ...<br />
        99.wav<br />

Run get_mfcc.py to get the MFCCs data<br />
or download it https://mycuhk-my.sharepoint.com/:x:/g/personal/1155158772_link_cuhk_edu_hk/EXTtCtUrgFFPqLltYS3saE8BBROPPZyGFp-NRz1ZWuRBSQ?e=FYoahq

Run model.py to train the model and get the model.pt

Run result_testing.py can return the predicted singer
(audio files in recordings are used for testing) ???

dataset.csv
dataset_features.py
dataset_preprocess.py
audio_segment.py
CNN.ipynb
are useless

Web
