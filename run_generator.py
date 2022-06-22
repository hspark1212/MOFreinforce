import json
import torch
from torch.utils.data import Dataset
from generator.stackRNN import StackAugmentedRNN
from generator.smiles_enumerator import SmilesEnumerator

ol_to_smiles = json.load(open("data/organiclinker_to_smiles.json"))
char_to_idx = json.load(open("data/char_to_idx.json"))


class OLDataset(Dataset):
    """
    Dataset for organic linker
    """

    def __init__(self, ol_to_smiles, augment=False):
        self.ol_name, self.smiles = zip(*ol_to_smiles.items())
        self.smiles_enumerator = SmilesEnumerator()
        self.augment = augment

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.augment:
            smiles = self.smiles_enumerator.randomize_smiles(self.smiles[idx])
        else:
            smiles = self.smiles[idx]
        inp = [char_to_idx["<"]] + [char_to_idx[s] for s in smiles] + [char_to_idx[">"]]
        return torch.LongTensor(inp)


def main():
    use_cuda = torch.cuda.is_available()
    dataset = OLDataset(ol_to_smiles, augment=False)
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    my_generator = StackAugmentedRNN(char_to_idx=char_to_idx, input_size=len(char_to_idx), hidden_size=hidden_size,
                                     output_size=len(char_to_idx), layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance, lr=lr)

    my_generator = my_generator.cuda()

    my_generator.fit(dataset, n_iterations=1000000, print_every=10000, save_dir="generator/saved_model")


if __name__ == "__main__":
    main()
