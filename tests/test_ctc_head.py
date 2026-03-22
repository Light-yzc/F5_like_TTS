import torch

from models.ctc_head import CTCAlignmentHead


def test_pack_target_hidden_states_moves_target_to_front():
    hidden_states = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ],
            [
                [11.0, 110.0],
                [12.0, 120.0],
                [13.0, 130.0],
                [14.0, 140.0],
                [15.0, 150.0],
            ],
        ]
    )
    target_mask = torch.tensor(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    packed_hidden, input_lengths = CTCAlignmentHead.pack_target_hidden_states(hidden_states, target_mask)

    expected = torch.tensor(
        [
            [
                [2.0, 20.0],
                [3.0, 30.0],
                [5.0, 50.0],
            ],
            [
                [11.0, 110.0],
                [14.0, 140.0],
                [0.0, 0.0],
            ],
        ]
    )

    assert input_lengths.tolist() == [3, 2]
    assert torch.equal(packed_hidden, expected)
