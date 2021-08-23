include("utils.jl")

# CONST
T        = 100  # maximum length
E        = 32  # embedding size
H        = 64  # hidden size
F        = 20  # number of filter
B        = 30  # batch_size
Ws       = [10,20,30]  # filters' size
EPOCH    = 50
FOLD     = 5
DROP     = 0.5
L1       = 0.0
L2       = 0.0
LR       = 1e-2
EPS      = 1e-5
SEED     = 1

zips(a::Vector{T}) where T = map(t-> map(x->x[t], a), 1:length(first(a)))

function leftpad(x::AbstractVector, p, t=length(x)+1; truncate=true)
  x = deepcopy(x)
  length(x) < t && (x = vcat(fill(p, t-length(x)), x))
  truncate && (x = x[end-t+1:end])
  x
end

JLD2.@load "./mimic3.pidxyy_unplanvdvp.jld2"
i = 100
x[i]
y[i]
xs,ys = x, y_unplan

ds, ps = zips(xs)
x = ds
join_episode(x) = vcat(split.(x)...)
remove_empty(x) = filter(x->!isempty(x), x)
x2 = @> x join_episode.() remove_empty.() leftpad.("PAD", 100)
x = x2
vx = @> vcat(x...) unique sort
X = length(vx)
Y = 2
PAD = 1
x = @> x indexin.([vx])
x = hcat(x...)
y = Int.(y_unplan) .+ 1

N = size(x)[end] ÷ B * B
x0, y0 = deepcopy(x), deepcopy(y)

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
          x -> reshape(x, (size(x)[1], size(x)[2], 1, size(x)[3])),
          [Chain(Conv((E, w), 1=>F, relu), pool1) for w∈Ws],
          xs -> vcat(squeezeall.(xs)...),
          Dense(length(Ws)*F, H, relu),
          Dense(H, Y),
         )
ps = m

loss(x, y) = nll(m(x), y)
loss(x, y)

optimizer=Adam()
ps = params(m)
for param in ps
  param.opt = deepcopy(optimizer)
end
for epoch = 1:EPOCH
  for (x,y)=dr
    J = @diff loss(x,y)
    for param in ps
      g = grad(J,param)
      Knet.update!(value(param),g,param.opt)
    end
  end
end

accuracy(m(xt), yt)

