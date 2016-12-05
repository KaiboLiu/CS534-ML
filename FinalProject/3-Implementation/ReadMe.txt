### ReadMe.txt#########

There Project includes 3 folders: 
   # Data,  includes all the source audio file we use.
   # Feature,  includes the feature files " train.dev & test.dev " we extracted from audio file
   # Result,   includes all the plot saving during the whole project.

For the files in it:
   # ExtractFeatures.py  % run 'python ExtractFeatures.py' to generate feature files from audio files.
   # diyDataset.py       % DIY the specificly subset from the whole wav file.
   # classifyMain.py     % run 'python classifyMain.py' to excute the classify training and testing process.
   # ClassifyModels.py   % called by classifyMain.py. It contains different classify models using in the project.
   # FeatureReduction.py % called by classifyMain.py. Working as the Feature Reduction Module.

   # cs534-final-project.pdf  % report file


To run the project,
   1. make sure the three folders are in the current directory.
   2. run 'diyDataset.py' (to build a specific training and testing dataset)
   2. run 'python ExtractFeatures.py' (to run this, please make sure you've installed 'librosa' library)
   3. run 'python classifyMain.py' (It's also okay to run this step directly, since we've saved two feature files in the folder.)