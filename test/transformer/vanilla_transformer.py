import torch
from pytest import approx

from src.transformer.vanilla_transformer import VanillaTransformer
from src.transformer.tokenizer import (
    vocab_size,
    encode_movechars,
)


class Test_VanillaTransformer:
    assert VanillaTransformer.__name__ == "VanillaTransformer"

    def test_prob_of_continuation(self):
        assert VanillaTransformer.prob_of_continuation.__name__ == "prob_of_continuation"

        model = VanillaTransformer(
            context_sz=1024,
            embd_dim=64,
            device="cpu",
            head_sz=1,
            lr=1e-5,
            num_heads=1,
            num_transformer_blocks=1,
            vocab_sz=vocab_size(),
        )

        prefix = encode_movechars("a4 e5 ").type(torch.long)
        continuation = encode_movechars("b4 Qxc4").type(torch.long).unsqueeze(0)
        prob = model.prob_of_continuation(prefix, continuation)
        assert 0.0 < prob < 1.0 or approx(prob) == 0.0 or approx(prob) == 1.0

        sum_prob = 0.0
        for i in range(vocab_size()):
            prob = model.prob_of_continuation(prefix, torch.tensor([[i]], dtype=torch.long))
            sum_prob += prob

        assert 1.0 == approx(sum_prob)
