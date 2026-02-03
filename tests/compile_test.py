import torch
from torch.linalg import vector_norm
import torch.utils.benchmark as benchmark
from torch.func import jvp, grad, vjp

torch_device = torch.device('cuda')
def norm_2(x, y):
    return torch.square(vector_norm(torch.abs(x) - y))
opt_norm = torch.compile(norm_2,backend='cudagraphs')

x_test = torch.randn(1024**2, dtype=torch.float, device=torch_device)

y_test = torch.randn(1024**2, device=torch_device)

def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

def hvp_norm(u, v): return hvp(lambda x: norm_2(x, y_test), (u,), (v,))

hvp_comp = torch.compile(hvp_norm, backend='inductor', mode='max-autotune')
#hvp_comp = torch.compile(hvp_norm, backend='cudagraphs')

v = torch.zeros_like(x_test)
v[0] = 1

for _ in range(3):
   hvp_comp(x_test, v)

t0 = benchmark.Timer(
    stmt='hvp_comp(x_test, v)',
    globals={'x_test': x_test, 'v': v, 'hvp_comp': hvp_comp})

t1 = benchmark.Timer(
    stmt='hvp_norm(x_test, v)',
    globals={'x_test': x_test, 'v': v, 'hvp_norm': hvp_norm})

print(t0.timeit(100))

print(t1.timeit(100))