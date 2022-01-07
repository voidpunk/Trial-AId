# imports
import torch
from torchdrug import data, datasets, core, models, tasks

# load SIDER dataset
dataset = datasets.SIDER(
    "./models/data/sider/",
    node_feature="pretrain",
    edge_feature="pretrain"
    )

# split the dataset into training, validation and test sets
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = data.ordered_scaffold_split(dataset, lengths)

# define the model
model = models.GIN(
    input_dim=dataset.node_feature_dim,
    hidden_dims=[
        1024,
        1024
    ],
    edge_input_dim=dataset.edge_feature_dim,
    short_cut=False,
    activation="sigmoid",
    concat_hidden=True,
    batch_norm=True,
    readout="mean"
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
    batch_size=512
    )

# load the pretrained infograph model
checkpoint = torch.load("./models/pretrain/infograph_unsupervised.pth")["model"]
task.load_state_dict(checkpoint, strict=False)

# train the model
solver.train(num_epoch=250)

# evaluate the model
metrics = solver.evaluate("valid")

# display the results
metrics_counter = 0
for key, value in metrics.items():
    metrics_counter += value.item()
print(round(metrics_counter/len(metrics)*100, 2), "%")


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
