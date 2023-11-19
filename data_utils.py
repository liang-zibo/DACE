from torch.utils.data import Dataset


# create dataset
class DACEDataset(Dataset):
    def __init__(self, seqs, attn_mask, loss_mask, run_times):
        self.seqs = seqs
        self.attn_mask = attn_mask
        self.loss_mask = loss_mask
        self.run_times = run_times

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            self.seqs[idx],
            self.attn_mask[idx],
            self.loss_mask[idx],
            self.run_times[idx],
        )
