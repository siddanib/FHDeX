import torch

# Look into Page 7 of https://arxiv.org/pdf/2412.06264
# A base class
class Flow_Base (torch.nn.Module):
    def __init__ (self):
        super().__init__()

    def step (self, x_t, c, t_start, t_end):
        h = t_end - t_start
        return x_t + h*self(x_t, c, t_start)

    def sample (self, x_0, c, n_steps=100):
        dt = 1.0/n_steps
        for i_t in range(n_steps):
            t_start = torch.ones(*x_0.shape[:-1],1)*i_t*dt
            t_end = t_start + dt
            x_0 = self.step(x_0, c, t_start, t_end)
        return x_0

class Flow_DeepONet (Flow_Base):
    def __init__ (self, output_size, kernel_size,
                 n_layers, layer_width, act_func = torch.nn.ReLU(),
                 residual_con=True, time_embedding=128,
                 normalize=True, first_kernel_size=2):
        super().__init__()
        self.input_size_1  = 1
        self.input_size_2  = 1
        self.input_size_3  = 1
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers    = n_layers
        self.layer_width = layer_width
        self.act_func    = act_func
        self.time_embedding = time_embedding
        self.normalize = normalize
        self.first_kernel_size = first_kernel_size

        self.module_list_1 = torch.nn.ModuleList([
                            building_block_cnn(self.input_size_1, self.layer_width,
                                           self.kernel_size, self.n_layers,
                                           act_func, residual_con, normalize,
                                           first_kernel_size)])

        self.module_list_2 = torch.nn.ModuleList([
                            building_block_cnn(self.input_size_2, self.layer_width,
                                           self.kernel_size, self.n_layers,
                                           act_func, residual_con, normalize,
                                           first_kernel_size)])

        self.module_list_3 = torch.nn.ModuleList([
                                SinusoidalTimeEmbedding(self.time_embedding)])
        self.module_list_3.append(building_block(self.time_embedding,self.layer_width,
                                           1,act_func,residual_con))

        for i in range(1, self.n_layers-1):
            self.module_list_3.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))

        self.combining_layer = building_block(self.layer_width,self.output_size,
                                           1,act_func,residual_con)

    def forward (self, x_t, c, t):
        for lyr in self.module_list_1:
            x_t = lyr(x_t)

        for lyr in self.module_list_2:
            c = lyr(c)

        for lyr in self.module_list_3:
            t = lyr(t)

        x_t = torch.transpose(x_t, -1, -2)
        c   = torch.transpose(c, -1, -2)

        x = self.combining_layer(x_t*c*t)
        x   = torch.transpose(x, -1, -2)
        return x

class building_block (torch.nn.Module):
    def __init__ (self, input_size, output_size, n_layers=1,
                  act_func = torch.nn.ReLU(), residual_con=True):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.act_func = act_func
        self.residual_con = residual_con

        self.module_list = torch.nn.ModuleList([
                            torch.nn.Linear(self.input_size,self.output_size,
                                            bias=True)])

        for i in range(1, self.n_layers):
            self.module_list.append(torch.nn.Linear(
                self.output_size,self.output_size,bias=True))

        self.module_list.append(torch.nn.Linear(self.output_size,
                                                self.output_size,bias=True))
        self.residual_layer = None
        if self.residual_con:
            self.residual_layer = torch.nn.Linear(self.input_size, self.output_size,
                                               bias=True)
    def forward (self, x):
        x1 = torch.clone(x)
        for lyr in self.module_list[:-1]:
            x1 = lyr(x1)
            x1 = self.act_func(x1)
        lyr = self.module_list[-1]
        x1 = lyr(x1)
        if self.residual_con:
            x1 += self.residual_layer(x)
        return x1

# Note that the current class only implemets periodic boundaries
class building_block_cnn (torch.nn.Module):
    def __init__ (self, input_size, output_size, kernel_size,
                  n_layers=1, act_func = torch.nn.ReLU(),
                  residual_con=True, normalize=True, first_kernel_size=2):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.act_func = act_func
        self.residual_con = residual_con
        self.normalize = normalize
        self.first_kernel_size = first_kernel_size

        self.module_list = torch.nn.ModuleList([
                            torch.nn.Conv1d(self.input_size,self.output_size,
                                            self.first_kernel_size, padding=0,
                                            bias=True)])
        if normalize:
            self.module_list.append(torch.nn.InstanceNorm1d(self.output_size,
                                                            affine=True))

        for i in range(1, self.n_layers):
            self.module_list.append(torch.nn.Conv1d(
                self.output_size, self.output_size,
                self.kernel_size, padding = "same",
                padding_mode="circular", bias=True))
            if normalize:
                self.module_list.append(
                        torch.nn.InstanceNorm1d(self.output_size,
                                                affine=True))

        # Final linear layer
        self.module_list.append(torch.nn.Conv1d(
                self.output_size, self.output_size,
                self.kernel_size, padding="same",
                padding_mode="circular", bias=True))

    def forward (self, x):
        x1 = torch.clone(x)
        # Apply the first convolutional layer separately
        lyr = self.module_list[0]
        pad_tpl = (int(self.first_kernel_size/2), 0)
        x1 = torch.nn.functional.pad(x1, pad_tpl, mode="circular")
        x1 = self.act_func(lyr(x1))
        ########################################################
        for lyr in self.module_list[1:-1]:
            x1 = lyr(x1)
            if lyr.__class__.__name__ == "Conv1d":
                if self.residual_con:
                    x1 += self.act_func(x1)
                else:
                    x1 = self.act_func(x1)
        lyr = self.module_list[-1]
        x1 = lyr(x1)
        return x1

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (..., 1)
        Returns:
            embedding: shape (..., embedding_dim)
        """
        half_dim = self.embedding_dim // 2
        # Compute the frequencies
        device = t.device
        exponents = torch.arange(half_dim, device=device) / half_dim
        ## Following https://github.com/cambridge-mlg/pdediff/blob/master/pdediff/nn/embedding.py#L9
        frequencies = 10000 ** exponents  # shape (half_dim,)
        # Compute embeddings
        args = t * frequencies
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

if __name__ == "__main__":
    mdl = Flow_DeepONet(1, 3, 2,64, act_func=torch.nn.ELU())
    x_t = torch.rand(5, 1, 10)
    c = torch.rand(5, 1, 10)
    t = torch.ones(5, 1, 1)
    print(mdl(x_t, c, t).shape)
    #print(mdl.step(x_t, c, t, t+0.005).shape)
    #print(mdl.sample(x_t, c).shape)
