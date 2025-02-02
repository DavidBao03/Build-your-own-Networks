import torch
import torch.nn as nn

class BasicExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.expert = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # shape: (batch_size, output_dim)
        return self.expert(x)

class BasicMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([BasicExpert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # shape: (batch_size, num_experts) -> (batch_size, 1, num_experts)
        expert_weight = self.gate(x).unsqueeze(1)
        print("expert_weight shape: ", expert_weight.shape)
        # shape: (batch_size, num_experts) -> (batch_size, 1, num_experts)
        expert_out_list = [expert(x).unsqueeze(1) for expert in self.experts]
        print("expert_out_list shape: ", expert_out_list[0].shape)
        # shape: (batch_size, num_experts, output_dim)
        expert_output = torch.cat(expert_out_list, dim=1)
        print("expert_output shape: ", expert_output.shape)
        # shape: (batch_size, output_dim)
        output = (expert_weight @ expert_output).squeeze(1)
        print("output shape: ", output.shape)
        return output
    
def test_basice_moe():
    batch_size = 2
    input_dim = 3
    output_dim = 4
    num_experts = 5
    model = BasicMoE(input_dim, output_dim, num_experts)
    x = torch.randn(batch_size, input_dim)
    y = model(x)
    print(y.shape)

test_basice_moe()