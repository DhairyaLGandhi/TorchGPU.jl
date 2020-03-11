module TorchGPU

greet() = print("Hello World!")

using Torch, CuArrays
using Cassette


function Torch.tensor(x::CuArray; dev = 0)
  Torch.from_blob(x, dev = dev)
end

Cassette.@context TorchCtx

using Torch.NNlib

function Cassette.overdub(ctx::TorchCtx, f, args...)
  f(Torch.to_tensor.(args)...)
end

function withtorch(f)
  ctx = TorchCtx()
  Cassette.overdub(ctx, f)
end

end # module
