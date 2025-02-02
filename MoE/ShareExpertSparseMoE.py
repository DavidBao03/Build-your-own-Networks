import torch
import torch.nn as nn
import torch.nn.functional as F

class MOEConfig:
    def __init__(self, hidden_dim, expert_num, topk, share_expert_num):
        self.hidden_dim = hidden_dim
        self.expert_num = expert_num
        self.topk = topk
        self.share_expert_num = share_expert_num

class BasicExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.expert = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # shape: (batch_size, output_dim)
        return self.expert(x)

class MOERounter(nn.Module):
    def __init__(self, hidden_dim, expert_num, topk):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_num)
        self.topk = topk
        self.expert_num = expert_num
    
    def forward(self, x): # x: (batch_size * seq_len, hidden_dim)
        # shape: (batch_size * seq_len, expert_num)
        router_logits = self.gate(x)
        # shape: (batch_size * seq_len, expert_num)
        router_probs = torch.softmax(router_logits, dim=-1)
        print("router_probs shape: ", router_probs.shape)
        print("router_probs: ", router_probs)
        # shape: (batch_size * seq_len, topk)
        router_weights, selected_experts = torch.topk(router_probs, self.topk, dim=-1)
        print("router_weights shape: ", router_weights.shape)
        print("router_weighrs: ", router_weights)
        print("selected_experts shape: ", selected_experts.shape)
        print("selected_experts: ", selected_experts)

        # Nomalize the router_weights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(x.dtype)

        # shape: (batch_size * seq_len, topk, expert_num)
        expert_mask = F.one_hot(selected_experts, num_classes=self.expert_num)
        print("expert_mask shape: ", expert_mask.shape)
        print("expert_mask: ", expert_mask)

        # shape: (expert_num, topk, batch_size * seq_len)
        expert_mask = expert_mask.permute(2, 1, 0)
        print("expert_mask shape: ", expert_mask.shape)
        print("expert_mask: ", expert_mask)

        return router_logits, router_weights, selected_experts, expert_mask
    
class SparseExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num
        self.topk = config.topk

        self.experts = nn.ModuleList([BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_num)])

        self.router = MOERounter(self.hidden_dim, self.expert_num, self.topk)
    
    def forward(self, x):
        # shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        # shape: (batch_size * seq_len, hidden_dim)
        hidden_state = x.view(-1, hidden_dim)

        router_logits, router_weights, selected_experts, expert_mask = self.router(hidden_state)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_state.dtype,
            device=hidden_state.device
        )

        for expert_idx in range(self.expert_num):
            expert = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx表示该专家作为top几来处理token
            # top_x表示该专家处理的token的位置
            # 如expert_idx = 0, idx = 0， top_x = 1 表示第1个专家作为top1处理第2个token
            print("idx: ", idx)
            print("top_x: ", top_x)

            # shape: (selected_token_number, hidden_dim)
            current_state = hidden_state.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)

            # shape: (selected_token_number, 1)
            current_hidden_states = expert(current_state) * router_weights[top_x, idx].unsqueeze(-1)

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_state.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits

class ShareExpertMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.moe_model = SparseExpert(config)
        self.shared_expert = nn.ModuleList(
            [BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.share_expert_num)]
        )

    def forward(self, x):
        sparse_moe_out, router_logits = self.moe_model(x)
        
        shared_expert_output = [
            expert(sparse_moe_out) for expert in self.shared_expert
        ]
        print("shared_expert_output shape: ", shared_expert_output[0].shape)
        shared_expert_output = torch.stack(shared_expert_output, dim=0)
        print("shared_expert_output shape: ", shared_expert_output.shape)
        shared_expert_output = shared_expert_output.sum(dim=0)
        print("shared_expert_output shape: ", shared_expert_output.shape)

        return shared_expert_output + sparse_moe_out, router_logits
    
def test_sparse_moe():
    batch_size = 2
    seq_len = 3
    hidden_dim = 4
    expert_num = 5
    topk = 3
    share_expert_num = 2
    config = MOEConfig(hidden_dim, expert_num, topk, share_expert_num)
    model = ShareExpertMoE(config)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y, router_logits = model(x)
    print(y.shape)
    print(router_logits.shape)

test_sparse_moe()
