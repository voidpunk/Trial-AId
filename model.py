# imports
from chembl_webresource_client.new_client import new_client
import torch
from torchdrug import data, models, tasks
from rdkit import Chem

# query the ChEMBL database for the molecule
query = "aspirin"
molecule = new_client.molecule
molecule_structure = molecule\
    .filter(molecule_synonyms__molecule_synonym__iexact=query)\
    .only("molecule_structures")

# retrieve the InChi
inchi = molecule_structure[0]["molecule_structures"]["standard_inchi"]
print(inchi)
print(type(inchi))

# construct the graph
graph = Chem.MolFromInchi(inchi)
print(graph)
print(type(graph))

# construct the torchdrug molecule object
graph = data.Molecule.from_molecule(graph, kekulize=True)
print(graph)
print(type(graph))

# # plot the graph
# # import matplotlib
# # import matplotlib.pyplot as plt
# # matplotlib.use("TkAgg")
# # graph.visualize()
# # plt.show()

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
checkpoint = torch.load("models/ClinTox/clintox_gin.pth")["model"]
task.load_state_dict(checkpoint, strict=False)

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

# predict the ClinTox features of the retrieved molecule
with torch.no_grad():
    model.eval()
    sample = data.graph_collate([{"graph": graph}])
    print(sample)
    print(sample["graph"].shape)
    # sample = utils.cuda(sample)
    pred = torch.sigmoid(task.predict(sample))
    print(pred)
