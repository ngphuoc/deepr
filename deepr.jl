begin
  using Ease  # include("./Ease/src/Ease.jl")
  using KnetModel   # include("./KnetModel/src/KnetModel")
  using Knet: randn!
  using Base.Iterators
  using CSV
  using DataFrames
  using EndpointRanges
  using Glob
  using GZip
  using Images
  using ImageView
  using JLD2
  using NMF
  using Plots
  using PyCall
  using Statistics
end

en = @env begin
  T        = 100  # maximum length
  E        = 32  # embedding size
  H        = 64  # hidden size
  F        = 20  # number of filter
  B        = 30  # batch_size
  EPOCH    = 50
  FOLD     = 5
  DROP     = 0.5
  L1       = 0.0
  L2       = 0.0
  LR       = 1e-2
  EPS      = 1e-5
  Ws       = [10,20,30]  # filters' size
  SEED     = 1
  atype    = KnetArray{Float32}
end

Knet.seed!(SEED)

JLD2.@load "./mimic3.pidxyy_unplanvdvp.jld2"
i = 100
x[i]
y[i]
xs,ys = x, y_unplan
ds, ps = zips(xs)
x = ds
@> length.(x) maximum, minimum, mean
join_episode(x) = vcats(split.(x))
remove_empty(x) = filter(x->!isempty(x), x)
x2 = @.> x join_episode remove_empty leftpad("PAD", 100)
x = x2
vx = @> x vcats unique sort
X = length(vx)
Y = 2
PAD = 1
@≥ x indexin.([vx]) cats
y = Int.(y_unplan) .+ 1

N = size(x)[end] ÷ B * B
x0, y0 = @.> x,y deepcopy

fold = 1
it = fold:5:N
ir = setdiff(1:N)
xr, yr = x0[:, ir], y0[ir]
xt, yt = x0[:, it], y0[it]
dr = minibatch(xr, yr, B)
dt = minibatch(xr, yr, B)
x,y = example = first(dr)

m = Chain(
          Embed(X, E),
          x -> dropout(x, DROP),
          x -> unsqueeze(x, dims=3),
          [Chain(Conv((E, w), 1=>F, relu), pool1) for w∈Ws],
          xs -> vcats(squeezeall.(xs)),
          Dense(length(Ws)*F, H, relu),
          Dense(H, Y),
         )
ps = m

loss(x, y) = nll(m(x), y)

@minibatch loss
nb = length(dr)
report = Report(1:1nb:nb*EPOCH, [d -> accuracy(m, d)], [dr, dt]);
train!(params(ps), loss, dr; report=report);
accuracy(m(xt), yt)

