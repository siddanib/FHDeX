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
        self.input_size_2  = 2
        self.input_size_3  = 1
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers    = n_layers
        self.layer_width = layer_width
        self.act_func    = act_func
        self.time_embedding = time_embedding
        self.normalize = normalize
        self.first_kernel_size = first_kernel_size

        self.module_list_2_down = torch.nn.ModuleList([
                                    building_block_cnn(self.input_size_2, self.layer_width,
                                           self.kernel_size, self.n_layers,
                                           act_func, residual_con, normalize,
                                           first_kernel_size)])

        self.module_list_2_up = torch.nn.ModuleList([
                                    building_block_cnn(2*self.layer_width, self.layer_width,
                                           self.kernel_size, self.n_layers,
                                           act_func, residual_con, normalize,
                                           0)])

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
        skip_connections = []
        x_t_c = torch.cat([x_t, c],dim=-2)
        for lyr in self.module_list_2_down:
            x_t_c = lyr(x_t_c)
            skip_connections.append(x_t_c)

        for ii, lyr in enumerate(self.module_list_2_up):
            skp_cnt = skip_connections[-(ii+1)]
            x_t_c = lyr(torch.cat([x_t_c,skp_cnt],dim=-2))

        for lyr in self.module_list_3:
            t = lyr(t)

        x_t_c = torch.transpose(x_t_c, -1, -2)

        x = self.combining_layer(x_t_c*t)
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

        if first_kernel_size > 0:
            self.module_list = torch.nn.ModuleList([
                            torch.nn.Conv1d(self.input_size,self.output_size,
                                            self.first_kernel_size, padding=0,
                                            bias=True)])
        else:
            self.module_list = torch.nn.ModuleList([
                            torch.nn.Conv1d(self.input_size,self.output_size,
                                            self.kernel_size, padding="same",
                                            padding_mode="circular", bias=True)])

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
        if self.first_kernel_size > 0:
            aa = int(self.first_kernel_size/2)
            pad_tpl = ( aa, aa-1)
            x1 = torch.nn.functional.pad(x1, pad_tpl, mode="circular")
        x1 = self.act_func(lyr(x1))
        ########################################################
        for lyr in self.module_list[1:-1]:
            x1 = lyr(x1)
            if lyr.__class__.__name__ == "Conv1d":
                if self.residual_con:
                    x1 += self.act_func(x1.clone())
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
    mdl = Flow_DeepONet(1, 3, 2, 64, act_func=torch.nn.ELU(),
                        first_kernel_size=8)
    x_t = torch.rand(1, 1, 10)
    c = torch.rand(1, 1, 10)
    t = torch.ones(1, 1, 1)
    mdl(x_t, c, t)
    #print(mdl(x_t, c, t))
    #x_t_flip = torch.flip(x_t, dims=[-1,])
    #c_flip = torch.flip(c, dims=[-1,])
    #print(mdl(x_t_flip, c_flip, t))
    ###print(mdl.step(x_t, c, t, t+0.005).shape)
    ###print(mdl.sample(x_t, c).shape)
    #print(mdl.sample(x_t, c))
    #print(mdl.sample(x_t_flip, c_flip))
