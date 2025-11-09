class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 600)
        self.ln1 = nn.LayerNorm(600)
        self.layer_2 = nn.Linear(600, 400)
        self.ln2 = nn.LayerNorm(400)
        self.layer_3 = nn.Linear(400, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.ln1(self.layer_1(s)))
        s = F.relu(self.ln2(self.layer_2(s)))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 600)
        self.layer_norm_1 = nn.LayerNorm(600)
        self.layer_2_s = nn.Linear(600, 400)
        self.layer_2_a = nn.Linear(action_dim, 400)
        self.layer_norm_2 = nn.LayerNorm(400)
        self.layer_3 = nn.Linear(400, 1)

        self.layer_4 = nn.Linear(state_dim, 600)
        self.layer_norm_4 = nn.LayerNorm(600)
        self.layer_5_s = nn.Linear(600, 400)
        self.layer_5_a = nn.Linear(action_dim, 400)
        self.layer_norm_5 = nn.LayerNorm(400)
        self.layer_6 = nn.Linear(400, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_norm_1(self.layer_1(s)))
        s1 = F.relu(self.layer_2_s(s1) + self.layer_2_a(a))
        s1 = self.layer_norm_2(s1)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_norm_4(self.layer_4(s)))
        s2 = F.relu(self.layer_5_s(s2) + self.layer_5_a(a))
        s2 = self.layer_norm_5(s2)
        q2 = self.layer_6(s2)

        return q1, q2