import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class AllenNLPSequential(torch.nn.Module):
    def __init__(
        self, moduleList, input_size, hidden_size, bidirectional, 
        residual_connection=False,
        dropout=0
    ):
        super(AllenNLPSequential, self).__init__()
        self.moduleList = moduleList
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_connection = residual_connection
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs, hidden_state):
        prev_outputs = (inputs, hidden_state)
        using_packed_sequence = isinstance(inputs, PackedSequence)
        num_modules = len(self.moduleList)
        residual_connection = self.residual_connection
        for i, module in enumerate(self.moduleList):
            outputs = module.forward(*prev_outputs)
            if using_packed_sequence:
                padded_output, seq_lengths = pad_packed_sequence(outputs[0], batch_first=True)
                # Do not apply dropout to the last layer
                if i != num_modules-1:
                    padded_output = self.dropout(padded_output)
                if residual_connection:
                    padded_prev_output, seq_lengths = pad_packed_sequence(prev_outputs[0], batch_first=True)
                    added_output = padded_output + padded_prev_output
                    outputs = (pack_padded_sequence(added_output, seq_lengths, batch_first=True),) + outputs[1:]
            else:
                if i != num_modules-1:
                    outputs = self.dropout(outputs)
                if residual_connection:
                    outputs = outputs + prev_outputs
            prev_outputs = outputs
        return outputs
