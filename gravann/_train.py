from torch import nn


def _weights_init(m):
    # Network initialization scheme (note that if xavier uniform is used all outputs will start at, roughly 0.5)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias.data, -0.0, 0.0)


def init_network(encoding, n_neurons=100, activation=nn.Sigmoid()):
    # Network architecture. Note that the dimensionality of the first linear layer must match the output
    # of the encoding chosen
    model = nn.Sequential(
        nn.Linear(encoding().dim, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, 1),
        activation,
    )

    # Applying our weight initialization
    _ = model.apply(_weights_init)

    return model


def train_on_batch(targets, labels, model, encoding, loss_fn, optimizer, scheduler, integrator, N):
    # Compute the loss (use N=3000 to start with, then, eventually, beef it up to 200000)
    predicted = integrator(targets, model, encoding, N=N)
    loss = loss_fn(predicted, labels)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Perform a step in LR scheduler to update LR
    scheduler.step(loss.item())

    return loss
