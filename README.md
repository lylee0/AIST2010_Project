# AIST2010_Project
Predicting Singers from Songs through Machine Learning

### Package Requirements
Please refer to the `requirements.txt` file for the necessary packages.

### Dataset Preparation
1. Download the dataset directory from the following link: [Dataset Link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155158772_link_cuhk_edu_hk/Egbl1_rh3NtGlnzfTuX5f2kBDVBFU-3HVHaV5hnBEZnyQw?e=u7x6Tl)

### MFCC Data Generation
1. Run the `get_mfcc.py` script to generate the MFCC (Mel-frequency cepstral coefficients) data.
   - Alternatively, you can download the pre-generated MFCC data from this link: [MFCC Data](https://mycuhk-my.sharepoint.com/:x:/g/personal/1155158772_link_cuhk_edu_hk/EXTtCtUrgFFPqLltYS3saE8BBROPPZyGFp-NRz1ZWuRBSQ?e=FYoahq)

### Model Training
1. Run the `model.py` script to train the model and generate the `model.pt` file.

### Result Testing
1. Run the `result_testing.py` script to obtain predictions for the singer.

### Web Application
#### Front-End:
- Open the HTML file located in the `front-end` directory using a live server.

#### Back-End:
- Run the `api.py` script to start the Flask Application for the backend.

**Note**: Due to limitations of the live server, it is recommended to run the front-end and back-end in separate windows of VSCode. If you encounter any issues with webpage refreshing, you can refer to the following Stack Overflow thread for potential solutions: [Preventing Page Refreshing](https://stackoverflow.com/questions/70435252/how-to-prevent-page-from-refreshing-after-javascript-fetch-post-request)
