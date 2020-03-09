module TorchGPU

greet() = print("Hello World!")

using Torch, CuArrays


function ATen.tensor(x::CuArray; dev = 0)
  ATen.from_blob(x, dev = dev)
end

end # module
