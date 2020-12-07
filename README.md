# classify-arabic-poetry
Code and dataset for building a machine learning model that classifies Arabic poetry according to the poetry meter.

This reposotory has the following files:

1. Arabic PCD version 2.0 (APCD2) dataset that consists of the following two files:
1.1. APCD_plus_porse_test.zip is the test set
1.2. APCD_plus_porse_test.zip is the train set

This dataset is based on APCD that has the following reference:
Yousef, W. A., Ibrahime, O. M., Madbouly, T. M., Mahmoud, M. A., El-Kassas, A. H., Hassan, A. O. Albohy, A. R., 2018. Poem Comprehensive Dataset (PCD). Available: https://hci-lab.github.io/ArabicPoetry-1-Private/#PCD

For a description of this dataset, refer to the following paper:
Gheith A. Abandah, Mohammed Z. Khedher, Mohammad R. Abdel-Majeed, Hamdi M. Mansour, Salma F. Hulliel, Lara M. Bisharat, 2020. Classifying and diacritizing Arabic poems using deep recurrent neural networks, Journal of King Saud University - Computer and Information Sciences, Accepted, Dec 2020.

2. Python program for training and testing an RNN model to classify Arabic poetry. Refer tot eh above paper for information about this model.
Classify_Poems_web.py

3. The APCD2 dataset in a format suitable for machine learning
APCD_plus_porse_all.zip

To execute the python program:
1) Download the python program in some folder.
2) Uncompress the file APCD_plus_porse_all.zip in the same folder.
3) Execute $python ./Classify_Poems_web.py

To cite this repository, use:
Gheith A. Abandah, 2020. Classify Arabic Poetry. GitHub; https://github.com/Gheith-Abandah/classify-arabic-poetry.git.

You can try the model trained to classify Arabic poems on the following page: http://www.abandah.com/gheith/Poetry/ .
