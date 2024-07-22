import numpy as np
import jax
import jax.numpy as jnp
jax.print_environment_info()
# 设置 JAX 全局设备为 CPU
# jax.config.update('jax_platform_name', 'cpu')

A = np.random.rand(300,300)
B = np.random.rand(300,300,4)

numpy_result_double        = np.einsum("ab,caP->cbP", A, B)
numpy_result_single        = np.einsum("ab,caP->cbP", A.astype(np.float32), B.astype(np.float32))

jax_result                 = jnp.einsum("ab,caP->cbP", jnp.array(A), jnp.array(B))

print(np.linalg.norm(numpy_result_double - numpy_result_single))
print(np.linalg.norm(         jax_result - numpy_result_single))
