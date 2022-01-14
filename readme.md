# Trial AId

[![](https://img.shields.io/badge/app-TrialAId-red.svg?&color=red&labelColor=grey&style=for-the-badge)](https://share.streamlit.io/arraymancer/trial-aid)


**TL;DR:**

Trial AId is a simple webapp that allows to search for a drug (experimental or not) and provides predictions through deep learning about the possible side effects and FDA approval of the drug, moreover it allows you to see the related clinical trials on that molecule currently available worldwide, all in one place and with one query.

**Description:**

Currently all the pieces of information provided to the subjects involved in clinical trials come from the preclinical phase on animal models. Often they are not enough detailed or accurate to reliably inform the patients about all the possible outcomes of the experimental treatment which the subject is going to receive.

This is where Trial AId comes: it is an AI-powered tool that aims at providing more detailed information to the patient about the possible outcomes of the clinical trial, in order to ***aid*** more informed decisions when choosing to participate to a ***trial***. This is possible thanks to a deep learning algorithm trained on hundreds of thousands of molecular structures and their corresponding properties and interactions.

## Technical description

Trial AId is based on the powerful Torchdrug library (written upon PyTorch) and the extensive ChEMBL database, containing more than 2.1 milions of chemical compounds. The clinical trials data is retrieved from ClinicalTrials.gov, an NIH website.

The model is pre-trained with unsupervised deep-learning on 250,000 molecules from the ZINC250k dataset.
After which, it is trained on thousands of molecules on 3 datasets: ClinTox, SIDER, and BBBP.
The model currently has a 90% accuracy on the ClinTox dataset,a 70% accuracy on selected tasks of the SIDER dataset (all the one shown in the app), and a 90% accuracy on the BBBP dataset.

The next goal is to pre-train the model on 2 million molecules from the ZINC2M dataset, to improve the overall performance of the model, and to train the side-effect prediction model jointly on the SIDER, OFFSIDES, MEDEFFECT, and FAERS datasets to greatly improve the accuracy of this model.

### To do:
- [x] ClinTox
- [x] SIDER
- [x] search for trial
- [x] pydeck
- [x] BBBP
- [ ] drug-drug interaction