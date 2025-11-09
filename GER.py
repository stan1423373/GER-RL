class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123, alpha=0.7, beta_start=0.2, beta_frames=300000, initial_td_error=100.0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)
        self.td_errors = deque(maxlen=buffer_size)  
        self.alpha = alpha  
        self.beta_start = beta_start 
        self.beta_frames = beta_frames 
        self.beta = beta_start 
        self.frame = 1  
        self.initial_td_error = initial_td_error 
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        self.buffer.append(experience)
        self.td_errors.append(self.initial_td_error) 
        self.count = min(self.count + 1, self.buffer_size)

    def size(self):
        return self.count

    def sample_batch(self, batch_size, greedy_prob=0.9):
        if self.count < batch_size:
            indices = range(self.count)
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)  
        else:
            td_errors = np.array(self.td_errors, dtype=np.float32)
            td_errors = np.maximum(td_errors, 1e-6)
            ranks = np.argsort(np.argsort(-td_errors))
            probabilities = (1.0 / (ranks + 1)) ** self.alpha
            probabilities /= probabilities.sum()  
            greedy_count = int(batch_size * greedy_prob)
            random_count = batch_size - greedy_count
            greedy_indices = np.random.choice(len(self.buffer), greedy_count, p=probabilities)
            random_indices = np.random.choice(len(self.buffer), random_count, replace=False)
            indices = np.concatenate((greedy_indices, random_indices))
        batch = [self.buffer[idx] for idx in indices]
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)    
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        be = self.beta
        fr = self.frame
        return s_batch, a_batch, r_batch, t_batch, s2_batch, indices, weights, be, fr

    def update_td_errors(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.td_errors[idx] = max(float(error), 1e-6)  

    def clear(self):
        self.buffer.clear()
        self.td_errors.clear()
        self.count = 0