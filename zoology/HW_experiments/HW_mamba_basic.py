from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig


config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=128, input_seq_len=64, kwargs={"num_kv_pairs": 4})],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=128, input_seq_len=64, kwargs={"num_kv_pairs": 4})],
    ),

    model = ModelConfig(
        block_type="MambaBlock",
        vocab_size=128,
        max_position_embeddings=0,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.mamba.Mamba",
            kwargs={"d_state": 16}),
        ),
        name="mamba",

)

configs = [config]