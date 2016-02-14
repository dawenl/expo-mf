# Experimental Results
By getting the data and running the following notebooks, you should be able to reproduce the exact results in the paper.

## Taste Profile Subset
- [processTasteProfile.ipynb](./processTasteProfile.ipynb): pre-process the data and create the train/test/validation splits.
- [ExpoMF_tasteProfileSub.ipynb](./ExpoMF_tasteProfileSub.ipynb): train the model and evaluate on the heldout test set.

## Gowalla (with location exposure covariates)
- [processGowalla.ipynb](./processGowalla.ipynb): pre-process the data and create the train/test/validation splits.
- [Location_ExpoMF_Gowalla.ipynb](./Location_ExpoMF_Gowalla.ipynb): train the model with location exposure covariates and evaluate on the heldout test set.
