module TorchGPU

greet() = print("Hello World!")

using Torch, CuArrays


function Torch.tensor(x::CuArray; dev = 0)
  Torch.from_blob(x, dev = dev)
end

Cassette.@context TorchCtx

using Torch.NNlib

function Cassette.overdub(ctx::TorchCtx, f, args...)
  f(Torch.to_tensor.(args)...)
end

end # module
