import numpy as np
import jax
import jax.numpy as jnp
# jax.print_environment_info()
# 设置 JAX 全局设备为 CPU
jax.config.update('jax_platform_name', 'gpu')

print(jax.config.jax_default_matmul_precision)
# jax.config.update('jax_default_matmul_precision', 'float32')
print(jax.config.jax_default_matmul_precision)
A = np.random.rand(300,300)
B = np.random.rand(300,300,4)

numpy_result_double        = np.einsum("ab,caP->cbP", A, B)
numpy_result_single        = np.einsum("ab,caP->cbP", A.astype(np.float32), B.astype(np.float32))

jax_result                 = jnp.einsum("ab,caP->cbP", jnp.array(A), jnp.array(B))

print(np.linalg.norm(numpy_result_double - numpy_result_single))
print(np.linalg.norm(         jax_result - numpy_result_single))

# 获取 jax.config 的所有属性
attributes = dir(jax.config)

# 打印每个属性及其值
for attribute in attributes:
    # 忽略以双下划线开头的特殊属性
    if not attribute.startswith('__'):
        try:
            value = getattr(jax.config, attribute)
            print(f'{attribute}: {value}')
        except Exception as e:
            print(f'Could not get value for {attribute}: {e}')