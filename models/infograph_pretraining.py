import torch
from torchdrug import core, datasets, tasks, models
import os

if not os.path.exists("./models/pretrain/infograph_unsupervised.pth"):

    dataset = datasets.ZINC250k(
        "./models/data/zinc250k/",
        node_feature="pretrain",
        edge_feature="pretrain"
        )

    gin_model = models.GIN(
        input_dim=dataset.node_feature_dim,
        hidden_dims=[
            512,
            512,
            512,
            512
        ],
        edge_input_dim=dataset.edge_feature_dim,
        batch_norm=True,
        readout="mean"
        )

    model = models.InfoGraph(
        gin_model,
        separate_model=False
        )

    task = tasks.Unsupervised(model)

    optimizer = torch.optim.Adam(
        task.parameters(),
        lr=1e-4
        )

    solver = core.Engine(
        task,
        dataset,
        None,
        None,
        optimizer,
        gpus=[0],
        batch_size=512
        )

    solver.train(num_epoch=100)
    solver.save("./models/pretrain/infograph_unsupervised.pth")
