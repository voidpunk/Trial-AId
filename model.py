# imports
from chembl_webresource_client.new_client import new_client
import torch
from torchdrug import data, models, tasks
from rdkit import Chem


def query(key: "str") -> "str":
    """
    Query the ChEMBL database for a molecule and return its InChI.
    """

    # query the ChEMBL database for the molecule
    molecule = new_client.molecule
    molecule_structure = molecule\
        .filter(molecule_synonyms__molecule_synonym__iexact=key)\
        .only("molecule_structures")

    # retrieve the InChi
    inchi = molecule_structure[0]["molecule_structures"]["standard_inchi"]
    print(inchi)
    # print(type(inchi))

    return inchi


def construct(inchi: "str") -> "graph":
    """
    Construct a molecular graph from an InChI.
    """

    # construct the graph
    graph = Chem.MolFromInchi(inchi)
    print(graph)
    # print(type(graph))

    # construct the torchdrug molecule object
    graph = data.Molecule.from_molecule(graph, kekulize=True)
    print(graph)
    # print(type(graph))
    return graph


def load():

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
        task=["FDA_APPROVED", "CT_TOX"],
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


def predict(graph, model, task) -> "dict":

    # predict the ClinTox features of the retrieved molecule
    with torch.no_grad():
        model.eval()
        sample = data.graph_collate([{"graph": graph}])
        print(sample)
        print(sample["graph"].shape)
        # sample = utils.cuda(sample)
        pred = torch.sigmoid(task.predict(sample))
        print(pred)
        # print(pred.shape)
        # print(pred[0][0].item())
        # print(pred[0][1].item())
        pred = {
            "FDA approved": round(pred[0][0].item()*100, 2),
            "toxic": round(pred[0][1].item()*100, 2)
        }

        return pred



if __name__ == "__main__":
    inchi = query("aspirin")
    graph = construct(inchi)
    model, task = load()
    pred = predict(graph, model, task)
    print("\n", pred)
else:
    pass


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
