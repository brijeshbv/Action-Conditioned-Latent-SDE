import torch
from torch import nn


class CDEFunc(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, action_shape):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.latent_size = latent_size
        self.action_shape = action_shape
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size , hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size * action_shape),
            nn.Tanh(),
        )

    def prod(self, t, z, dXdt):
        assert t.shape == ()
        vector_field = self.forward(t, z)
        out = (vector_field @ dXdt.unsqueeze(-1)).squeeze(-1)
        return out

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.f_net(z)
        z = z.view(z.size(0), self.latent_size, self.action_shape)
        return z

class CDEFuncPost(torch.nn.Module):
    def __init__(self, latent_size, context_size, hidden_size, action_shape):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFuncPost, self).__init__()
        self.latent_size = latent_size
        self.action_shape = action_shape
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size,  latent_size * action_shape),
            nn.Tanh(),
        )

    def prod(self, t, z, dXdt):
        assert t.shape == ()
        vector_field = self.forward(t, z)
        out = (vector_field @ dXdt.unsqueeze(-1)).squeeze(-1)
        return out

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.f_net(z).view(z.size(0), self.latent_size, self.action_shape)
        return z

class VectorField(torch.nn.Module):
    def __init__(self, X, func, is_tensor, is_prod):
        super(VectorField, self).__init__()

        self.X = X
        self.func = func
        self.is_tensor = is_tensor
        self.is_prod = is_prod

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)

        if self.is_prod:
            # out is of shape (..., hidden_channels)
            out = self.func.prod(t, z, control_gradient)
        else:
            # vector_field is of shape (..., hidden_channels, input_channels)
            vector_field = self.func(t, z)
            if self.is_tensor:
                # out is of shape (..., hidden_channels)
                # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
                out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
            else:
                out = tuple((vector_field_ @ control_gradient_.unsqueeze(-1)).squeeze(-1)
                            for vector_field_, control_gradient_ in zip(vector_field, control_gradient))

        return out

    # torchsde backend
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)
