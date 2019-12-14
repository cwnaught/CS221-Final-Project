This is the Github page to our CS221 Project: Frame Perfect - An Auteur Classifier.
Note that the data has not been uploaded due to file size limits. If you would like to see it to run our code, please email thomlins@stanford.edu.

Individual pre-processing files:
- FullProcess.py
- FullProcessConcat.py
- baselineProcessing.py
- transferProcess.py

Each file inputs .mp4 located in a specific directory and processes them into either TFFs or frame volumes. Each file is read then the desired frames are stored to be saved at the end.


Classifiers:
- baseline.py (SVM)
- BW CNN.ipynb
- Color CNN Flow.ipynb
- concat CNN.ipynb

Each of these files correspond to a specific CNN (or SVM). They each input pre-processed data that is pre-separated into training, validation, and testing folders.