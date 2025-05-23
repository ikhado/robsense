# Reference: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py

class TransformerConfig:
    def __init__(self, mlp_dim, num_heads, num_layers, attention_dropout_rate, dropout_rate, embedding_dim):
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim


class ModelConfig:
    def __init__(self, model_name: str, patches: int, hidden_size: int, transformer_config: TransformerConfig,
                 classifier='token',
                 representation_size=None):
        self.model_name = model_name
        self.patches = patches
        self.hidden_size = hidden_size
        self.transformer_config = transformer_config
        self.classifier = classifier
        self.representation_size = representation_size


def get_model_config(model_size='small'):
    config_map = {
        'small': get_s16_config,
        'base': get_b16_config,
        'large': get_l16_config
    }

    if model_size in config_map:
        return config_map[model_size]()
    else:
        raise ValueError("Invalid model size. Choose from 'small', 'base', or 'large'.")


def get_s16_config():
    """Returns the ViT-S/16 configuration."""
    hidden_size = 384
    transformer_config = TransformerConfig(
        mlp_dim=1536,
        num_heads=6,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.0,
        embedding_dim=hidden_size
    )
    return ModelConfig(
        model_name='ViT-S_16',
        patches=16,
        hidden_size=hidden_size,
        transformer_config=transformer_config
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    hidden_size = 768
    transformer_config = TransformerConfig(
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.0,
        embedding_dim=hidden_size
    )
    return ModelConfig(
        model_name='ViT-B_16',
        patches=16,
        hidden_size=hidden_size,
        transformer_config=transformer_config
    )


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    hidden_size = 1024
    transformer_config = TransformerConfig(
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        embedding_dim=hidden_size
    )
    return ModelConfig(
        model_name='ViT-L_16',
        patches=16,
        hidden_size=hidden_size,
        transformer_config=transformer_config
    )
