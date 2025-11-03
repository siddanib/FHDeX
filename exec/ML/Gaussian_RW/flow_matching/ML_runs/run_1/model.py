import torch

# Look into Page 7 of https://arxiv.org/pdf/2412.06264
# A base class
class Flow_Base (torch.nn.Module):
    def __init__ (self):
        super().__init__()

    def step (self, x_t, c, t_start, t_end):
        h = t_end - t_start
        return x_t + h*self(x_t+h*0.5*self(x_t,c,t_start),c,t_start+0.5*h)

    def sample (self, x_0, c, n_steps=100):
        dt = 1.0/n_steps
        for i_t in range(n_steps):
            t_start = torch.ones_like(x_0)*i_t*dt
            t_end = t_start + dt
            x_0 = self.step(x_0, c, t_start, t_end)
        return x_0

# Conditional flow matching
class Flow_MLP (Flow_Base):
    def __init__ (self,cond_size, output_size,
                 n_layers, layer_width, act_func = torch.nn.ReLU(),
                 residual_con=True):
        super().__init__()
        self.input_size  = output_size + cond_size + 1
        self.output_size = output_size
        self.n_layers    = n_layers
        self.layer_width = layer_width
        self.act_func    = act_func

        self.module_list = torch.nn.ModuleList([
                            building_block(self.input_size,self.layer_width,
                                           1,act_func,residual_con)])

        for i in range(1, self.n_layers-1):
            self.module_list.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))

        self.module_list.append(
                            building_block(self.layer_width,self.output_size,
                                           1,act_func,residual_con))

    def forward (self, x_t, c, t):
        x = torch.cat([x_t, c, t],dim=-1)
        for lyr in self.module_list:
            x = lyr(x)
        return x

class Flow_DeepONet (Flow_Base):
    def __init__ (self,cond_size, output_size,
                 n_layers, layer_width, act_func = torch.nn.ReLU(),
                 residual_con=True, time_embedding=128):
        super().__init__()
        self.input_size_1  = output_size
        self.input_size_2  = cond_size
        self.input_size_3  = 1
        self.output_size = output_size
        self.n_layers    = n_layers
        self.layer_width = layer_width
        self.act_func    = act_func
        self.time_embedding = time_embedding

        self.module_list_1 = torch.nn.ModuleList([
                            building_block(self.input_size_1,self.layer_width,
                                           1,act_func,residual_con)])

        self.module_list_2 = torch.nn.ModuleList([
                            building_block(self.input_size_2,self.layer_width,
                                           1,act_func,residual_con)])

        self.module_list_3 = torch.nn.ModuleList([
                                SinusoidalTimeEmbedding(self.time_embedding)])
        self.module_list_3.append(building_block(self.time_embedding,self.layer_width,
                                           1,act_func,residual_con))

        for i in range(1, self.n_layers-1):
            self.module_list_1.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))
            self.module_list_2.append(
                            building_block(self.layer_width,self.layer_width,
                                           1,act_func,residual_con))
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
        x = self.combining_layer(x_t*c*t)
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
                self.layer_width,self.output_size,bias=True))

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

class Hierarchical_Model (Flow_Base):
    def __init__ (self, history_length, cond_size, output_size,
                 n_layers, layer_width, act_func = torch.nn.ReLU(),
                 residual_con=True, time_embedding=128):
        super().__init__()
        self.history_length = history_length
        self.hierarchy_levels = torch.nn.ModuleList()
        self.max_level = 0
        for ilev in range(history_length+1):
            self.hierarchy_levels.append(
                    Flow_DeepONet((ilev+1)*cond_size, output_size,
                                  n_layers, layer_width,
                                  act_func, residual_con,
                                  time_embedding))

    # This defines which levels need to be trained
    def train_levels(self,levels_list):
        for ilev in range(self.history_length+1):
            if ilev in levels_list:
                self.hierarchy_levels[ilev].train(True)
                for param in self.hierarchy_levels[ilev].parameters():
                    param.requires_grad = True
            else:
                self.hierarchy_levels[ilev].train(False)
                for param in self.hierarchy_levels[ilev].parameters():
                    param.requires_grad = False

    def forward (self, x_t, c, t):
        # The condition to be passed
        cond = torch.narrow(c, -1, -2, 2)
        x = self.hierarchy_levels[0](x_t,cond,t)
        for ilev in range(1, self.max_level+1):
            cond = torch.narrow(c, -1, -2*(ilev+1), 2*(ilev+1))
            x += self.hierarchy_levels[ilev](x_t,cond,t)

        return x

if __name__ == "__main__":
    #mdl = Flow_DeepONet(2,1,2,64, act_func=torch.nn.ELU())
    #x = torch.tensor([[0.5, 0.83]])
    #t = torch.tensor([[0.1]])
    #print(mdl(t, x, t))
    #print(mdl.step(t, x, t,t+0.005))
    #print(mdl.sample(t, x))
    #embedding_layer = SinusoidalTimeEmbedding(embedding_dim=64)
    #time_tensor = torch.tensor([[0.0], [0.5], [1.0]])  # Shape: (3, 1)
    #embeddings = embedding_layer(time_tensor)

    #print(embeddings.shape)  # Expected output: (3, 8)
    #print(embeddings)


    mdl = Hierarchical_Model(1, 2,1,2,64)
    levels_list = [0,1]
    mdl.train_levels(levels_list)
    x = torch.tensor([[0.5, 0.83, 0.5, 0.83]])
    t = torch.tensor([[0.1]])
    print(mdl(t, x, t))
    mdl.max_level = 1
    print(mdl(t, x, t))
    print(mdl.sample(t,x))
