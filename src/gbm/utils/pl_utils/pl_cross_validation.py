class CV:
    def __init__(self, n_splits, shuffle, random_state):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in kf.split(X, y):
            yield train_index, test_index

    def get_n_splits(self):
        return self.n_splits

    def __repr__(self):
        return f'CV(n_splits={self.n_splits}, shuffle={self.shuffle}, random_state={self.random_state})'