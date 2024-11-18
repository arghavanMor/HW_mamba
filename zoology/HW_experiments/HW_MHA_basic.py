from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig


config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=128, input_seq_len=64, kwargs={"num_kv_pairs": 4})],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=128, input_seq_len=64, kwargs={"num_kv_pairs": 4})],
    ),
    model=ModelConfig(
        vocab_size=128,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),

)

configs = [config]