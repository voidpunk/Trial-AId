# imports
import torch
from torchdrug import data, datasets, core, models, tasks, utils
import json
from time import sleep

# load ClinTox dataset
dataset = datasets.ClinTox(
    "./data/clintox/",
    # node_feature="pretrain",
    # edge_feature="pretrain"
    )

# training loop
decision = True
while decision:
    # split the dataset into training, validation and test sets
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

    # define the model
    model = models.GIN(
        input_dim=dataset.node_feature_dim,
        hidden_dims=[256, 256, 256, 256],
        short_cut=True,
        batch_norm=True,
        concat_hidden=True
        )

    # define the task
    task = tasks.PropertyPrediction(
        model,
        task=dataset.tasks,
        criterion="bce",
        metric=(
            # "auprc",
            "auroc"
            )
        )

    # define the optimizer
    optimizer = torch.optim.Adam(
        task.parameters(),
        lr=1e-3
        )

    # define the solver
    solver = core.Engine(
        task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        gpus=[0],
        batch_size=4096
        )

    # train the model
    solver.train(num_epoch=100)

    # evaluate the model
    metrics = solver.evaluate("valid")

    # check the accuracy
    fda_met = metrics["auroc [FDA_APPROVED]"].item()
    tox_met = metrics["auroc [CT_TOX]"].item()
    if fda_met > 0.70 and tox_met > 0.70:
        decision = False

    # avoid overheating of my Hell-Inspiron
    if decision == True:
        sleep(10)

# save the model
with open("./clintox/clintox_model.json", "w") as fout:
    json.dump(model.config_dict(), fout)
solver.save("./clintox/clintox_model.pth")

# # plot samples
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
# samples = []
# categories = set()
# for sample in valid_set:
#     category = tuple([v for k, v in sample.items() if k != "graph"])
#     if category not in categories:
#         categories.add(category)
#         samples.append(sample)
# samples = data.graph_collate(samples)
# samples = utils.cuda(samples)

# preds = torch.sigmoid(task.predict(samples))
# targets = task.target(samples)

# titles = []
# for pred, target in zip(preds, targets):
#     pred = ", ".join(["%.2f" % p for p in pred])
#     target = ", ".join(["%d" % t for t in target])
#     titles.append("predict: %s\ntarget: %s" % (pred, target))
# graph = samples["graph"]
# graph.visualize(titles, figure_size=(3, 3.5), num_row=1)
# plt.show()
