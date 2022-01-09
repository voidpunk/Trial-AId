# imports
from chembl_webresource_client.new_client import new_client
import torch
from torchdrug import data, models, tasks
from rdkit import Chem


def query(key: "str") -> "tuple(str, dict)":
    """
    Query the ChEMBL database for a molecule and return its InChI.
    """

    # query the ChEMBL database for the molecule
    molecule = new_client.molecule\
        .filter(molecule_synonyms__molecule_synonym__iexact=key)[0]
        # .only("molecule_structures")[0]

    # print("\n\n", molecule, "\n\n")

    # retrieve the infos
    inchi = molecule["molecule_structures"]["standard_inchi"]
    # print(inchi)
    # print(type(inchi))
    first_approval = molecule["first_approval"]
    # print(first_approval)
    indication_class = molecule["indication_class"]
    # print(indication_class)
    max_phase = molecule["max_phase"]
    # print(max_phase)
    natural_product = molecule["natural_product"]
    # print(natural_product)
    oral = molecule["oral"]
    # print(oral)
    parenteral = molecule["parenteral"]
    # print(parenteral)
    topical = molecule["topical"]
    # print(topical)

    infos = {
        "first_approval": first_approval,
        "indication_class": indication_class,
        "max_phase": max_phase,
        "natural_product": natural_product,
        "oral": oral,
        "parenteral": parenteral,
        "topical": topical
    }

    return inchi, infos


def construct(inchi: "str") -> "graph":
    """
    Construct a molecular graph from an InChI.
    """

    # construct the graph
    graph = Chem.MolFromInchi(inchi)
    # print(graph)
    # print(type(graph))

    # construct the torchdrug molecule object
    graph = data.Molecule.from_molecule(graph, kekulize=True)
    # print(graph)
    # print(type(graph))
    return graph


def load_clintox():

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
            # "auprc",
            "auroc"
            )
        )

    # load the weights and task settings
    checkpoint = torch.load("models/clintox/clintox_model.pth")["model"]
    task.load_state_dict(checkpoint, strict=False)

    return model, task


def load_sider():

    # define the model
    model = models.GIN(
        input_dim=69,
        hidden_dims=[256, 256, 256, 256],
        # edge_input_dim=11,
        num_mlp_layer=2,
        short_cut=False,
        batch_norm=True,
        concat_hidden=True,
        # readout="mean",
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
            # "auprc",
            "auroc"
            )
        )

    # load the weights and task settings
    checkpoint = torch.load("models/sider/sider_model.pth")["model"]
    task.load_state_dict(checkpoint, strict=False)

    return model, task


def predict(graph, model, task, dataset) -> "dict":

    # predict the ClinTox features of the retrieved molecule
    with torch.no_grad():

        model.eval()
        sample = data.graph_collate([{"graph": graph}])
        # print(sample)
        # print(sample["graph"].shape)
        # sample = utils.cuda(sample)

        pred = torch.sigmoid(task.predict(sample))
        # print(pred)
        # print(pred.shape)
        # print(pred[0][0].item())
        # print(pred[0][1].item())

        if dataset == "clintox":
            pred = {
                "FDA approval": round(pred[0][0].item()*100, 2),
                "toxicity": round(pred[0][1].item()*100, 2)
            }

        elif dataset == "sider":
            pred = {
            "Hepatobiliary disorders": round(pred[0][0].item()*100, 2),
            "Metabolism & nutrition disorders": round(pred[0][1].item()*100, 2),
            # "Product issues": round(pred[0][2].item()*100, 2),
            "Eye disorders": round(pred[0][3].item()*100, 2),
            # "Investigations": round(pred[0][4].item()*100, 2),
            "Musculoskeletal & connective tissue disorders": round(pred[0][5].item()*100, 2),
            "Gastrointestinal disorders": round(pred[0][6].item()*100, 2),
            # "Social circumstances": round(pred[0][7].item()*100, 2),
            "Immune system disorders": round(pred[0][8].item()*100, 2),
            "Reproductive system & breast disorders": round(pred[0][9].item()*100, 2),
            "Neoplasms benign, malignant & unspecified": round(pred[0][10].item()*100, 2),
            # "General disorders & administration site conditions": round(pred[0][11].item()*100, 2),
            "Endocrine disorders": round(pred[0][12].item()*100, 2),
            "Surgical & medical procedures": round(pred[0][13].item()*100, 2),
            "Vascular disorders": round(pred[0][14].item()*100, 2),
            "Blood & lymphatic system disorders": round(pred[0][15].item()*100, 2),
            "Skin & subcutaneous tissue disorders": round(pred[0][16].item()*100, 2),
            "Congenital, familial & genetic disorders": round(pred[0][17].item()*100, 2),
            "Infections & infestations": round(pred[0][18].item()*100, 2),
            "Respiratory, thoracic & mediastinal disorders": round(pred[0][19].item()*100, 2),
            "Psychiatric disorders": round(pred[0][20].item()*100, 2),
            "Renal & urinary disorders": round(pred[0][21].item()*100, 2),
            "Pregnancy, puerperium & perinatal conditions": round(pred[0][22].item()*100, 2),
            "Ear & labyrinth disorders": round(pred[0][23].item()*100, 2),
            "Cardiac disorders": round(pred[0][24].item()*100, 2),
            "Nervous system disorders": round(pred[0][25].item()*100, 2),
            "Injury, poisoning & procedural complications": round(pred[0][26].item()*100, 2)
            }

        return pred


eval1 = False
eval2 = True

if __name__ == "__main__":

    if eval1 == True:
        inchi, _ = query("aspirin")
        graph = construct(inchi)
        model, task = load_clintox()
        pred = predict(graph, model, task, "clintox")
        print("\nClinTox:", pred)
        model, task = load_sider()
        pred = predict(graph, model, task, "sider")
        print("\nSIDER:", pred)

    if eval2 == True:
        from torchdrug import datasets
        dataset = datasets.SIDER("./data/sider/")
        model, task = load_sider()
        batch = data.graph_collate(dataset[:10])
        pred = task.predict(batch)
        print(batch)
        print(pred)


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
