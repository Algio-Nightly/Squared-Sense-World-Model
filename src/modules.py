import jax
import jax.numpy as jnp
from flax import nnx
import optax
from einops import rearrange


class SEBlock3D(nnx.Module):
    def __init__(self, channels: int, reduction: int = 4, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(channels, channels // reduction, rngs=rngs)
        self.fc2 = nnx.Linear(channels // reduction, channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        w = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        w = nnx.sigmoid(self.fc2(nnx.gelu(self.fc1(w))))
        return x * w

class Siamese3DCNN(nnx.Module):

    def __init__(
            self, in_channels: int = 32, 
            hidden_dim: int = 64, 
            out_dim: int = 128, 
            *, 
            rngs: nnx.Rngs
        ):
        self.conv1 = nnx.Conv(in_channels, hidden_dim, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME", rngs=rngs)
        self.gn1 = nnx.GroupNorm(num_features=hidden_dim, num_groups=8, rngs=rngs)
        self.se1 = SEBlock3D(channels=hidden_dim, rngs=rngs) # Feature Compressor

        self.conv2 = nnx.Conv(in_features= hidden_dim, out_features=out_dim, kernel_size=(3,3,3), strides=(2, 2, 2), padding="SAME", rngs=rngs)
        self.gn2 = nnx.GroupNorm(num_features=out_dim, num_groups=8, rngs=rngs)
        self.se2 = SEBlock3D(out_dim, rngs=rngs)

    def __call__(self, shell:jax.Array) -> jax.Array:
        x = self.se1(nnx.gelu(self.gn1(self.conv1(shell))))
        x = self.se2(nnx.gelu(self.gn2(self.conv2(x))))

        return rearrange(x, "b, d, h, w, c -> b (d h w) c") 


class Positional3DEncoding(nnx.Module):
        
    ...

class MemoryModule(nnx.Module):

    ...

