from chembl_webresource_client.new_client import new_client
import pubchempy as pcp
import torch
from torchdrug import data, models, tasks, datasets, utils
from rdkit import Chem
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def query(key):
    client = new_client.molecule
    # sure checks:
    key = key.strip().replace('"', '').replace("'", "")
    if key[:6].upper() == "CHEMBL":
        molecule = client.filter(chembl_id=key)[0]
    elif key[:7].lower() == "inchi=":
        inchikey = pcp.get_compounds(key, namespace="inchi")[0].inchikey
        molecule = client.filter(molecule_structures__standard_inchi_key=inchikey)[0]
    elif key.isdigit():
        compounds = pcp.get_compounds(int(key), namespace="cid")
        inchikey = compounds[0].inchikey
        molecule = client.filter(molecule_structures__standard_inchi_key=inchikey)[0]
    elif key[:5].lower() == "key:":
        key_mod = key[4:]
        molecule = client.filter(molecule_structures__standard_inchi_key=key_mod)[0]
    else:
        try:
            # unsure checks:
            try:
                compounds = pcp.get_compounds(key, namespace="name")
            except pcp.BadRequestError:
                print("\nNAME\n")
                pass
            if len(compounds) == 0:
                try:
                    compounds = pcp.get_compounds(key, namespace="smiles")
                except pcp.BadRequestError:
                    print("\nSMILES\n")
                    pass
            if len(compounds) == 0:
                try:
                    compounds = pcp.get_compounds(key, namespace="inchikey")
                except pcp.BadRequestError as e:
                    print("\nINCHIKEY\n")
                    print(e)
                    pass
            if len(compounds) == 0:
                try:
                    compounds = pcp.get_compounds(key, namespace="formula")
                except pcp.BadRequestError:
                    print("\nFORMULA\n")
                    pass
            if len(compounds) == 0:
                try:
                    compounds = pcp.get_compounds(key, namespace="inchi")
                except pcp.BadRequestError as e:
                    print("\nINCHI\n")
                    print(e)
                    pass
            # if len(compounds) == 0:
            #     # not working and I don't know why
            #     try:
            #         key_mod = f"InChI={key}"
            #         compounds = pcp.get_compounds(key_mod, namespace="inchi")
            #     except pcp.BadRequestError as e:
            #         print("\nINCHI2\n")
            #         print(e)
            #         pass
            if len(compounds) != 0:
                inchikey = compounds[0].inchikey
                molecule = client.filter(molecule_structures__standard_inchi_key=inchikey)[0]
            else:
                molecule = None
        except pcp.TimeoutError:
            molecule = None
    # last check:
    if molecule is None:
        molecule = client.filter(molecule_synonyms__molecule_synonym__iexact=key)[0]
        print("\nNAME2\n")
        if molecule is None:
            molecule = client.filter(molecule_structures__standard_inchi_key=key)[0]
            print("\nINCHIKEY2\n")
        if molecule is None:
            molecule = client.filter(smiles=key, similarity=70)[0]
            print("\nSMILES2\n")
        if molecule is None:
            return None
    else:
        return molecule


def extract_data(molecule, type="inchi"):
    """
    Given a molecule and its infos as dictionary, it extracts and reorders
    the information which is returned as a tuple o either InChI or SMILES, and
    the other infos.
    """
    # retrieve the infos
    inchi = molecule["molecule_structures"]["standard_inchi"]
    smiles = molecule["molecule_structures"]["canonical_smiles"]
    first_approval = molecule["first_approval"]
    indication_class = molecule["indication_class"]
    max_phase = molecule["max_phase"]
    natural_product = molecule["natural_product"]
    oral = molecule["oral"]
    parenteral = molecule["parenteral"]
    topical = molecule["topical"]
    pref_name = molecule["pref_name"]
    # organize the infos
    infos = {
        "first_approval": first_approval,
        "indication_class": indication_class,
        "max_phase": max_phase,
        "natural_product": natural_product,
        "oral": oral,
        "parenteral": parenteral,
        "topical": topical,
        "pref_name": pref_name
    }
    # return the infos
    if type == "smiles":
        return smiles, infos
    elif type == "inchi":
        return inchi, infos


def construct(mol, type="inchi", pretrain="True"):
    """
    Given an InChI or SMILES, it constructs its graph and returns it as a
    torchdrug molecule object.
    """
    # construct the RDKit molecule object
    if type == "inchi":
        # construct the graph
        graph = Chem.MolFromInchi(mol)
    elif type == "smiles":
        # construct the graph
        graph = Chem.MolFromSmiles(mol)
    # construct the torchdrug molecule object
    if pretrain:
        graph = data.Molecule.from_molecule(
            graph,
            kekulize=True,
            node_feature="pretrain",
            edge_feature="pretrain"
            )
    else:
        graph = data.Molecule.from_molecule(
            graph,
            kekulize=True,
            node_feature="default",
            edge_feature="default"
            )
    # return the graph object
    return graph


def load_clintox():
    """
    It loads the ClinTox pretrained model and task.
    """
    # define the model
    model = models.GIN(
        input_dim=69,
        hidden_dims=[256, 256, 256, 256],
        short_cut=True,
        batch_norm=True,
        concat_hidden=True
        )
    # define the task
    task = tasks.PropertyPrediction(
        model,
        task=[
            "FDA_APPROVED",
            "CT_TOX"
            ],
        criterion="bce",
        metric=(
            "auroc"
            )
        )
    # load the weights and task settings
    checkpoint = torch.load("models/clintox/clintox_model.pth", map_location=device)["model"]
    task.load_state_dict(checkpoint, strict=False)
    # return the model and task
    return model, task


def load_sider():
    """
    It loads the SIDER pretrained model and task.
    """
    # define the model
    model = models.GIN(
        input_dim=22,
        hidden_dims=[
            512,
            512,
            512,
            512,
            512,
            512,
            512,
            512
        ],
        edge_input_dim=11,
        num_mlp_layer=2,
        short_cut=False,
        batch_norm=True,
        concat_hidden=True,
        readout="mean",
        activation="sigmoid"
        )
    # define the task"auroc"
    task = tasks.PropertyPrediction(
        model,
        task=[
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications",
    ],
        criterion="bce",
        metric=(
            "auroc"
            )
        )
    # load the weights and task settings
    checkpoint = torch.load("models/sider/sider_model.pth", map_location=device)["model"]
    task.load_state_dict(checkpoint, strict=False)
    # return the result
    return model, task


def load_bbbp():
    """
    It loads the BBBP pretrained model and task.
    """
    # define the model
    model = models.GIN(
        input_dim=22,
        hidden_dims=[
            512,
            512,
            512,
            512
        ],
        edge_input_dim=11,
        num_mlp_layer=2,
        activation="sigmoid",
        short_cut=False,
        batch_norm=True,
        concat_hidden=True,
        readout="mean",
        )
    # define the task
    task = tasks.PropertyPrediction(
        model,
        task=[
            "p_np",
    ],
        criterion="bce",
        metric=(
            "auroc"
            )
        )
    # load the weights and task settings
    checkpoint = torch.load("models/bbbp/bbbp_model.pth", map_location=device)["model"]
    task.load_state_dict(checkpoint, strict=False)
    # return the result
    return model, task


def predict(graph, model, task, dataset, func="sigmoid", df=False):
    """
    Given a graph, a model (among the ones in this file), a task, and a dataset,
    it predicts the labels according to the model.
    """
    # predict the ClinTox features of the retrieved molecule
    with torch.no_grad():
        model.eval()
        sample = data.graph_collate([{"graph": graph}])
        # print(sample)
        # print(sample["graph"].shape)
        # sample = utils.cuda(sample)
        pred_raw = task.predict(sample)
        if func == "sigmoid":
            pred = torch.sigmoid(pred_raw)
        elif func == "softmax":
            pred = torch.softmax(pred_raw, dim=1)
        elif func == "argmax":
            pred = torch.argmax(pred_raw, dim=1)
        elif func == "none":
            pred = pred_raw
        elif func == "step":
            pred = torch.where(pred_raw > 0.5, torch.ones_like(pred_raw), torch.zeros_like(pred_raw))
        # print(pred)
        # print(pred.shape)
        # print(pred[0][0].item())
        if dataset == "clintox":
            pred = {
                "FDA approval": round(pred[0][0].item()*100, 2),
                "toxicity": round(pred[0][1].item()*100, 2)
            }
        elif dataset == "sider":
            pred = {
            "Neurologic\t\t\t\t\t\t\t\t": round(pred[0][25].item()*100, 2),                                             # 86.88
            "Hematology\t\t\t\t\t\t\t\t": round(pred[0][15].item()*100, 2),                                             # 74.03
            "Infections & infestations": round(pred[0][18].item()*100, 2),                              # 71.69
            "Endocrine\t\t\t\t\t\t\t\t": round(pred[0][12].item()*100, 2),                                              # 69.71
            "Hepatobiliary\t\t\t\t\t\t\t\t": round(pred[0][0].item()*100, 2),                                           # 69.32
            "Reproductive\t\t\t\t\t\t\t\t": round(pred[0][9].item()*100, 2),                                            # 69.15
            "Psychiatric\t\t\t\t\t\t\t\t": round(pred[0][20].item()*100, 2),                                            # 69.06
            # "Investigations": round(pred[0][4].item()*100, 2),                                        # 68.55
            "Urologic\t\t\t\t\t\t\t\t": round(pred[0][21].item()*100, 2),                                                # 68.09
            "Gastroenterologic\t\t\t\t\t\t\t\t": round(pred[0][6].item()*100, 2),                                        # 66.99
            "Vascular\t\t\t\t\t\t\t\t": round(pred[0][14].item()*100, 2),                                               # 67.62
            "Oncologic\t\t\t\t\t\t\t\t": round(pred[0][10].item()*100, 2),                                              # 66.74
            "Ophtalmic\t\t\t\t\t\t\t\t": round(pred[0][3].item()*100, 2),                                               # 65.04
            "Product issues": round(pred[0][2].item()*100, 2),                                          # 65.00
            # --------------------------------------
            # "Musculoskeletal & connective tissue disorders": round(pred[0][5].item()*100, 2),         # 62.60
            # "Metabolism & nutrition disorders": round(pred[0][1].item()*100, 2),                      # 61.39
            # "Injury, poisoning & procedural complications": round(pred[0][26].item()*100, 2),         # 61.32
            # "Cardiac disorders": round(pred[0][24].item()*100, 2),                                    # 60.22
            # "Ear & labyrinth disorders": round(pred[0][23].item()*100, 2),                            # 60.11
            # "Immune system disorders": round(pred[0][8].item()*100, 2),                               # 59.04
            # "Social circumstances": round(pred[0][7].item()*100, 2),                                  # 58.02
            # "General disorders & administration site conditions": round(pred[0][11].item()*100, 2),   # 60.68
            # "Surgical & medical procedures": round(pred[0][13].item()*100, 2),                        # 50.11
            # "Skin & subcutaneous tissue disorders": round(pred[0][16].item()*100, 2),                 # 50.03
            # "Congenital, familial & genetic disorders": round(pred[0][17].item()*100, 2),             # 50.68
            # "Respiratory, thoracic & mediastinal disorders": round(pred[0][19].item()*100, 2),        # 54.99
            # "Pregnancy, puerperium & perinatal conditions": round(pred[0][22].item()*100, 2),         # 56.20
            }
        elif dataset == "bbbp":
            pred = {
            "BBB penetration": round(pred[0][0].item()*100, 2),
            }
        if df:
            return pd.DataFrame.from_dict(pred, orient="index", columns=["score"])
        else:
            return pred


def get_info_n_pred(key_mol, from_key=False):
    """
    A single function that wraps up all the process of the functions in this file
    in order to make it easier to use the module.
    """
    if from_key:
        # query the database
        molecule = query(key_mol)
    else:
        molecule = key_mol
    # extract the data
    inchi, infos = extract_data(molecule)
    # construct the molecules
    graph_default = construct(inchi, pretrain=False)
    graph_pretrain = construct(inchi, pretrain=True)
    # load the models and tasks
    clintox_model, clintox_task = load_clintox()
    sider_model, sider_task = load_sider()
    bbbp_model, bbbp_task = load_bbbp()
    # get the predictions
    clintox_pred = predict(graph_default, clintox_model, clintox_task, dataset="clintox")
    sider_pred = predict(graph_pretrain, sider_model, sider_task, dataset="sider")
    bbbp_pred = predict(graph_pretrain, bbbp_model, bbbp_task, dataset="bbbp", func="step")
    # return the prediction
    return {
        "infos": infos,
        "graph": graph_default,
        "clintox_pred": clintox_pred,
        "sider_pred": sider_pred,
        "bbbp_pred": bbbp_pred
    }



eval1 = False
eval2 = True

if __name__ == "__main__":

    if eval1 == True:
        inchi, _ = query("aspirin")
        graph = construct(inchi)
        # model, task = load_clintox()
        # pred = predict(graph, model, task, "clintox")
        # print("\nClinTox:", pred)
        model, task = load_sider()
        pred = predict(graph, model, task, "sider")
        print("\nSIDER:", pred)

    if eval2 == True:
        # PLEASE IGNORE THIS MESS D:
        dataset = datasets.SIDER(
            "./models/data/sider/",
            node_feature="pretrain",
            edge_feature="pretrain"
            )
        # print(dataset[0]["graph"])
        # quit()
        model, task = load_sider()

        # working code:
        # batch = data.graph_collate(dataset[0:1])
        # print(batch)
        # print(type(batch))
        # print(len(batch))
        # pred = task.predict(batch)

        # now fucking working code:
        # graph = dataset[0]["graph"]
        # batch = data.graph_collate([{"graph": graph}])
        # print(batch)
        # print(type(batch))
        # print(len(batch))
        # pred = task.predict(batch)

        # now fucking working code:
        mol, _ = query("aspirin", "smiles")
        graph = construct(mol, "smiles")
        batch = data.graph_collate([{"graph": graph}])
        print(batch)
        print(type(batch))
        print(len(batch))
        pred = task.predict(batch)

        print(pred)


# Junk code:

# # plot the graph
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# graph.visualize()
# plt.show()


# # check the model of the ClinTox dataset
# dataset = datasets.ClinTox(
#     "./molecule-datasets/",
#     # node_feature="pretrain",
#     # edge_feature="pretrain"
#     )
# for i in range(25):
#     with torch.no_grad():
#         model.eval()
#         sample = data.graph_collate([dataset[i]])
#         # print(sample)
#         print(sample["graph"].shape)
#         # sample = utils.cuda(sample)
#         pred = torch.sigmoid(task.predict(sample))
#         print(pred)
#         print(task.target(sample))
