# classify-arabic-poetry
Code and dataset for building a machine learning model that classifies Arabic poetry according to the poetry meter.

This reposotory has the following files:

1. Arabic PCD version 2.0 (APCD2) dataset that consists of the following two files:
(a) APCD_plus_porse_test.zip is the test set
(b) APCD_plus_porse_test.zip is the train set

This dataset is based on APCD that has the following reference:
Yousef, W. A., Ibrahime, O. M., Madbouly, T. M., Mahmoud, M. A., El-Kassas, A. H., Hassan, A. O. Albohy, A. R., 2018. Poem Comprehensive Dataset (PCD). Available: https://hci-lab.github.io/ArabicPoetry-1-Private/#PCD

For a description of this dataset, refer to the following paper:
G.A. Abandah, M.Z. Khedher, M.R. Abdel-Majeed, H.M. Mansour, S.F. Hulliel, L.M. Bisharat, Classifying and diacritizing Arabic poems using deep recurrent neural networks, Journal of King Saud University – Computer and Information Sciences, https://doi.org/10.1016/j.jksuci.2020.12.002, Dec 2020.

2. Python program for training and testing an RNN model to classify Arabic poetry. Refer tot eh above paper for information about this model.
Classify_Poems_web.py

3. The APCD2 dataset in a format suitable for machine learning
APCD_plus_porse_all.zip

To execute the python program:
(a) Download the python program in some folder.
(b) Uncompress the file APCD_plus_porse_all.zip in the same folder.
(c) Execute $python ./Classify_Poems_web.py

4. The paper that describes this work.

To cite this repository, use:
Gheith A. Abandah, 2020. Classify Arabic Poetry. GitHub; https://github.com/Gheith-Abandah/classify-arabic-poetry.git.

You can try the model trained to classify Arabic poems on the following page: http://www.abandah.com/gheith/Poetry/ .
