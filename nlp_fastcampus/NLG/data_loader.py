import os
from re import M
from torchtext.legacy import data, datasets

class DataLoader:

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=256,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False
                ):

        super().__init__()

        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if dsl else None,
            eos_token='<EOS>' if dsl else None
        )

        self.tgt = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if use_bos else None,
            eos_token='<EOS>' if use_eos else None
        )

        if train_fn is not None and valid_fn is not None and exts is not None:
            train = data.TabularDataset(
                path=train_fn,
                format='tsv',
                fields=[('src', self.src), ('tgt', self.tgt)],
            )
            valid = data.TabularDataset(
                path=valid_fn,
                format='tsv',
                fields=[('src', self.src), ('tgt', self.tgt)],
            )

            self.train_iter = data.BucketIterator(
                train,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=shuffle,
                sort_key=lambda x: len(x.tgt) + (max_length + len(x.src)),
                sort_within_batch=True,
            )
            self.valid_iter = data.BucketIterator(
                valid,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=False,
                sort_key=lambda x: len(x.tgt) + (max_length + len(x.src)),
                sort_within_batch=True,
            )

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


