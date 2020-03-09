module TorchGPU

greet() = print("Hello World!")

using Torch, CuArrays


function Torch.tensor(x::CuArray; dev = 0)
  Torch.from_blob(x, dev = dev)
end

end # module
