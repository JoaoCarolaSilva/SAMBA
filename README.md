# SAMBA
This repository contains the source code used to develop SAMBA (Synthetic  Assemblies for Microbiome Based Agriculture)

Backbone (Please run in the following order

Additional_Feature_Testing.py - generates and screens triplet features
Classifier_Stage1_Selection.py - compares models, then selects the resampling strategy and number of features
Classifier_Stage2_Optuna_SHAP.py - tunes and evaluates the final model, ranks strains by P_Bemeficial and runs SHAP
Classifier_model_Syncom.py - assembles Syncoms using the avaiabble apprroaches

If you would like to run app.py using your results, please replace the files in the models folder, with the one generated on your run.

Deployment

app.py - FastAPI web app. You can run it and then use the interface by accessing localhost:8000 or alternatively https://samba.itqb.unl.pt/ can be used to access the web version. 
