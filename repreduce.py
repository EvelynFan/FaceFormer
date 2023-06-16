import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer = nn.Linear(128, 23370*3)
    def forward(self, x):
        return self.layer(x)


def time_model(model):
    with profile(activities=[ProfilerActivity.CPU],
      with_stack=True,
      experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
      x = torch.rand(1, 278, 128)
      model(x)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))
    
def main():
    model = Net()
    print("regular model")
    print(model)
    time_model(model)
    model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized
    print("int8 model")
    print(model_int8)
    time_model(model_int8)
main()


