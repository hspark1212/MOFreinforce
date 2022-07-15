import json
import os

import torch
from torch.utils.data import Dataset
from generator.stackRNN import StackAugmentedRNN
from generator.smiles_enumerator import SmilesEnumerator
import selfies as sf

import codecs
from SmilesPE.tokenizer import SPE_Tokenizer

# 0. character embedding
ol_to_smiles = json.load(open("data/v3/ol_to_smiles.json"))
# char_to_idx = json.load(open("data/char_to_idx.json"))

# 1. bpe
# vocab_to_idx = json.load(open("data/vocab_to_idx.json"))
# pretrained_tokenizer = codecs.open("data/tokenizer/tokenized_smiles.txt")
# spe = SPE_Tokenizer(pretrained_tokenizer)

# 2. selfies
ol_to_selfies = json.load(open("data/v3/ol_to_selfies.json"))
vocab_to_idx = json.load(open("data/v3/vocab_to_idx.json"))


class OLDataset(Dataset):
    """
    Dataset for organic linker
    """

    def __init__(self, ol_to_smiles, ol_to_selfies=None, augment=False):
        if ol_to_selfies is not None:
            self.ol_name, self.selfies = zip(*ol_to_selfies.items())
        self.ol_name, self.smiles = zip(*ol_to_smiles.items())

        self.smiles_enumerator = SmilesEnumerator()
        self.augment = augment

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        # 0. character embedding
        if self.augment:
            smiles = self.smiles_enumerator.randomize_smiles(self.smiles[idx])
        else:
            smiles = self.smiles[idx]
        inp = [char_to_idx["<"]] + [char_to_idx[s] for s in smiles] + [char_to_idx[">"]]
        """

        """
        # 1. bpe
        tokenized_smiles = spe.tokenize(smiles).split()
        inp = [char_to_idx["<"]] + [vocab_to_idx[s] for s in tokenized_smiles] + [char_to_idx[">"]]
        """
        # 2. selfies
        sf_ = self.selfies[idx]

        encoded = sf.selfies_to_encoding(selfies=sf_, vocab_stoi=vocab_to_idx, enc_type="label")
        inp = [1] + encoded + [2]

        return torch.LongTensor(inp)


def main():
    use_cuda = torch.cuda.is_available()
    dataset = OLDataset(ol_to_smiles, ol_to_selfies=ol_to_selfies, augment=False)
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    my_generator = StackAugmentedRNN(vocab_to_idx=vocab_to_idx,
                                     input_size=len(vocab_to_idx), hidden_size=hidden_size,
                                     output_size=len(vocab_to_idx), layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance, lr=lr)

    my_generator = my_generator.cuda()


    save_dir = "generator/saved_model_selfies_2/"
    os.makedirs(save_dir, exist_ok=True)
    my_generator.load_model("generator/saved_model_selfies/generator_970000.ch")
    my_generator.fit(dataset, n_iterations=1000000, print_every=10000, save_dir=save_dir)


if __name__ == "__main__":
    main()
