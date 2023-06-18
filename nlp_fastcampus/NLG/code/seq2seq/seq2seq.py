from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Attention(nn.Module):
    # attention = Q, K, V
    # w = softmax(Q.W.K_T)
    # c = W.V

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (bs, length, hidden_size), Encoder's hs
        # |h_t_tgt| = (bs, 1, hidden_size), Decoder's hs
        # |mask| = (bs, length)

        # Q.W
        query = self.linear(h_t_tgt)
        # |query| = (bs, 1, hidden_size)

                                        # 1, 2번을 transpose (length, hs)
        weight = torch.bmm(query, h_src.transpose(1, 2))
        # |weight| = (bs, 1, length)

        if mask is not None :
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))

        weight = self.softmax(weight)

        context_vector = torch.bmm(weight, h_src)
        # |context_vector| = (bs, 1, hidden_size)

        return context_vector


class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
    

    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_size)

        if isinstance(emb, tuple):
            x, lengths = emb
            # pack_padded_sequence 타입으로 변환
            # rnn에서 <PAD>가 포함된 데이터를 더 잘 처리
            # ex.
            # [1, 2, 3]  => [1, 2, 3,
            # [3, 4]         3, 4, _]
            x= pack(x, lengths.tolist(), batch_first=True)
            # x = PackedSequence(data=tensor([1, 3, 2, 4, 3]), batch_size=tensor([2, 2, 1]))

        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (bs, length, hs / 2 * 2)
        # |h[0]| = (num_layers * 2, bs, hs / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layer=4, dropout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
                           # input feeding sizeW
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=n_layer,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True
        )

    
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (bs, 1, word_vec_size)
        # |h_t_1_tilde| = (bs, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)

        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # If this is the First time-step
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)

        y, h = self.rnn(x, h_t_1)

        return y, h


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, x):
        # |x| = (bs, length, hidden_size)

        y = self.softmax(self.output(x))
        # |y| = (bs, length, output_size)

        return y


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size,
            hidden_size,
            n_layers=n_layers, dropout_p=dropout_p
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layer=n_layers, dropout_p=dropout_p
        )
        self.attention = Attention(hidden_size)
        
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    
    def generate_mask(self, x, length):
        # |length| = (bs, )
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]

            else :
                mask += [x.new_ones(1, l).zero_()]
        
        mask = torch.cat(mask, dim=0).bool()

        return mask


    # Encoder hiddens를 Decoder hiddens로 변환
    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        hiddens, cells = encoder_hiddens
        # |hiddens| = (n_layers * 2, bs, hs / 2)

        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)]
            # |new_hiddens| = (bs, hs)
            new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
        # |new_hiddens| = (n_layers, bs, hs)

        return (new_hiddens, new_cells)


    def fast_merge_encoder_hiddnens(self, encoder_hiddens):
        # from (n_layers * 2 , bs, hs / 2) to (n_layers, bs, hs)

        h_0_tgt, c_0_tgt = encoder_hiddens
        # |h_0_tgt| = (n_layers * 2, bs, hs/2)

        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size).transpose(0, 1).contiguous()
        # |h_0_tgt.tranpose(0, 1)| = (bs, n_layers * 2, hs/2)
        # contiguous() = 메모리에 잘 붙어있게?
        # |.view| = (bs, -1, hs) = (bs, n_layers, hs)
        # |.transpose(0, 1)| = (n_layers, bs, hs)
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size,
                                                            -1,
                                                            self.hidden_size).transpose(0, 1).contiguous()

        return h_0_tgt, c_0_tgt                                                        


    # 학습 부분
    def forward(self, src, tgt):
        # |src| = (bs, length)
        # |tgt| = (bs, length2, |v2|)
        # |output| = (bs, length2, |v2|)
        batch_size = tgt.size(0)

        mask = None
        x_length = None

        if isinstance(src, tuple):
            x, x_length = src

            mask = self.generate_mask(x, x_length)
            # |mask| = (bs, length)

        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        emb_src = self.emb_src(x)
        # |emb_src| = (bs, length, word_vec_size)

        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # encoder hidden_state, encoder 마지막 step hidden_state
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        h_0_tgt = self.fast_merge_encoder_hiddnens(h_0_tgt)
        # |h_0_tgt| = ((n_layers, bs, hs), cell_State)
        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (bs, length2, word_Vec_size)

        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt

        for t in range(tgt.size(1)):

            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            # |emb_t| = (bs, 1(unsqueeze), word_Vec_size)

            decoder_output, decoder_hidden = self.decoder(emb_t,
                                                          h_t_tilde,
                                                          decoder_hidden
                                                         )
            # |decoder_output| = (bs, 1, hs)
            # |decoder_hidden| = (n_layers, bs, hs)
            context_vector = self.attention(h_src, decoder_output, mask)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
                                                         context_vector
                                                        ], dim=-1)))
            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        # |h_tilde| = (bs, length2, hs)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (bs, length2, output_size)

        return y_hat


    # def search(self, src, is_greedy=True, max_length=255):
    #     mask, x_length = None, None

    #     if isinstance(src, tuple):
    #         x, x_length = src
    #         mask = self.generate_mask(x, x_length)

    #     else:
    #         x = src
    #     batch_size = x.size(0)

    #     emb_src = self.emb_src(x)
    #     h_src, h_0_tgt = self.encoder((emb_src, x_length))
    #     h_0_tgt = self.fast_merge_encoder_hiddnens(h_0_tgt)

    #     y = x.new(batch_size, 1).zero_() # + data_loader.BOS
    #     is_decoding = x.new_ones(batch_size, 1).bool()
    #     decoder_hidden = h_0_tgt
    #     h_t_tilde, y_hats, indice = None, [], []

    #     while is_decoding.sum() > 0 and len(indice) < max_length:

    #         emb_t = self.emb_dec(y)

    #         decoder_output, decoder_hidden = self.decoder(emb_t,
    #                                                       h_t_tilde,
    #                                                       decoder_hidden
    #                                                      )
    #         context_vector = self.attention(h_src, decoder_output, mask)
    #         h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output,
    #                                                      context_vector
    #                                                     ], dim=-1)))

    #         y_hat = self.generator(h_t_tilde)

    #         y_hats += [y_hat]

    #         if is_greedy:
    #             y = torch.topk(y_hat, 1, dim=-1)[1].squeeze(-1)
    #         else:
    #             y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

    #         y = y.masked_fill(~is_decoding) # , data_loader.PAD)
    #         is_decoding = is_decoding * torch.ne(y) # , data_loader.EOS

    #         indice += [y]

    #     y_hats = torch.cat(y_hats, dim=1)
    #     indice = torch.cat(indice, dim=-1)

    #     return y_hats, indice