import torch
from torchdrug import core, datasets, tasks, models


dataset = datasets.ZINC250k(
    "./models/data/zinc250k/",
    node_feature="pretrain",
    edge_feature="pretrain"
    )

model = models.GIN(
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

task = tasks.AttributeMasking(
    model,
    mask_rate=0.15
    )

optimizer = torch.optim.Adam(
    task.parameters(),
    lr=1e-3
    )

solver = core.Engine(
    task,
    dataset,
    None,
    None,
    optimizer,
    gpus=[0],
    batch_size=256
    )

solver.train(num_epoch=100)

solver.save("./models/pretrain/attributemasking_unsupervised.pth")