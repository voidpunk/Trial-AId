
# Trial AId

[![](https://img.shields.io/badge/app-TrialAId-red.svg?&color=red&labelColor=grey&style=for-the-badge)](https://share.streamlit.io/voidpunk/trial-aid)

![](https://img.shields.io/website?url=https%3A%2F%2Fshare.streamlit.io%2Fvoidpunk%2Ftrial-aid&style=flat-square)
![](https://img.shields.io/github/v/release/voidpunk/Trial-AId?include_prereleases&style=flat-square)
![](https://img.shields.io/badge/Python-3.8-blue?style=flat-square)

**TL;DR:**
<p style="text-align: justify;">
Trial AId is a webapp that allows to search for a drug (experimental or not) and provides predictions through deep learning about the possible side effects, and the probability of: FDA approval, toxicity, penetration of the BBB, and product issues. Moreover, it allows to see the related clinical trials on that molecule currently available worldwide, all in one place and with one single query.
</p>

**Description:**
<p style="text-align: justify;">
Currently all the pieces of information provided to the subjects involved in clinical trials come from the preclinical phase on in vitro and animal models. Often they are not enough detailed or accurate to reliably inform the patients about all the possible outcomes of the experimental treatment they are going going to receive.
<br>
This is where Trial AId comes: it is an AI-powered tool that aims at providing more detailed and reliable information to the patient about the possible outcomes of the clinical <b>trial</b>, in order to <b>aid</b> more informed decisions when choosing to participate to a trial. This is possible thanks to a deep learning algorithm trained on hundreds of thousands of molecules, their properties and interactions.
</p>

## About

<p style="text-align: justify;">
Trial AId is based on the powerful Torchdrug library (written upon PyTorch) and two enormous databases: PubChem (110+ millions of compounds) and ChEMBL (2.1+ millions of compounds). The clinical trials data is retrieved from ClinicalTrials.gov (400,000+ studies from 220 countries).
The model is pre-trained with unsupervised deep-learning on 250,000 molecules from the ZINC250k dataset. After that, it is specifically trained on thousands of molecules on 3 datasets: ClinTox, SIDER, and BBBP.
Currently the model has a 90% accuracy on the ClinTox dataset,a 70% accuracy on selected tasks of the SIDER dataset (all the one shown), and a 90% accuracy on the BBBP dataset.
<br><br>
The next goals are to pre-train the model on 2 million molecules from the ZINC2M dataset, to improve the overall performance of the model, and to train the side-effect prediction model jointly on the SIDER, OFFSIDES, MEDEFFECT, and FAERS datasets to greatly improve the accuracy of this model. Currently I am working on elaborating the FAERS dataset, the biggest dataset of drug side-effects, provided by the FDA.
</p>

### To do:
- [ ] drug-drug interaction
- [ ] pretrain on ZINC2M
- [ ] FAERS dataset
- [ ] MEDEFFECT dataset
- [ ] OFFSIDES dataset