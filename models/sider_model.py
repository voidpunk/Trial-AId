# imports
import torch
from torchdrug import data, datasets, core, models, tasks, utils
import json
from time import sleep

# load SIDER dataset
dataset = datasets.SIDER(
    "./data/sider/",
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
        lr=1e-4
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
    solver.train(num_epoch=200)

    # evaluate the model
    metrics = solver.evaluate("valid")

    # check the accuracy
    metrics_counter = 0
    for key, value in metrics.items():
        # print(key, value)
        metrics_counter += value.item()
    if metrics_counter >= len(metrics) * 0.7:
        decision = False

    # avoid overheating of my Hell-Inspiron
    if decision == True:
        sleep(10)

# save the model
with open("./sider/sider_model.json", "w") as fout:
    json.dump(solver.config_dict(), fout)
solver.save("./sider/sider_model.pth")

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
