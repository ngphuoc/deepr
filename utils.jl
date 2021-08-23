using MacroTools: postwalk, striplines, isexpr, @forward, @capture, animals, striplines
using Flux: @functor, nfan, glorot_uniform, glorot_normal
using Random

"Fast zip+splat"
zips(a::Vector{T}) where T = map(t-> map(x->x[t], a), 1:length(first(a)))

zips(a::Vector{<:Tuple}) = tuple(map(t-> map(x->x[t], a), 1:length(first(a)))...)

zips(a::T) where {T<:Tuple} = map(t-> map(x->x[t], a), 1:length(first(a)))

""" cat+splat

    julia> cats([rand(1, 2) for _=1:3], dims=1)
    3×2 Matrix{Float64}:
     0.701238  0.704747
     0.464978  0.769693
     0.800235  0.947824
"""
cats(xs; dims=ndims(xs[1])+1) = cat(xs..., dims=dims)

macro extract(m, vs)
  rhs = Expr(:tuple)
  for v in vs.args
    push!(rhs.args, :($m.$v))
  end
  ex = :($vs = $rhs) |> striplines
  esc(ex)
end

"""
Extends Lazy's @>
"""
macro >(exs...)
  @assert length(exs) > 0
  callex(head, f, x, xs...) = ex = :_ in xs ? Expr(:call, Expr(head, f, xs...), x) : Expr(head, f, x, xs...)
  thread(x) = isexpr(x, :block) ? thread(rmlines(x).args...) : x
  thread(x, ex) =
    if isexpr(ex, :call, :macrocall)
      callex(ex.head, ex.args[1], x, ex.args[2:end]...)
    elseif isexpr(ex, :tuple)
      Expr(:tuple,
           map(ex -> isexpr(ex, :call, :macrocall) ?
               callex(ex.head, ex.args[1], x, ex.args[2:end]...) :
               Expr(:call, ex, x), ex.args)...)
    elseif @capture(ex, f_.(xs__))
      :($f.($x, $(xs...)))
    elseif isexpr(ex, :block)
      thread(x, rmlines(ex).args...)
    else
      Expr(:call, ex, x)
    end
  thread(x, exs...) = reduce(thread, exs, init=x)
  esc(thread(exs...))
end

"Extend @>"
macro >=(x, exs...)
  esc(macroexpand(Main, :($x = @> $x $(exs...))))
end
# alias
var"@≥" = var"@>="

sampleμσ(μ::T, σ::AbstractArray{T}) where T = μ .+ oftype(σ, randn(Float32, size(σ))) .* σ

sampleμσ(μ::AbstractArray{T}, σ::T) where T = μ .+ oftype(μ, randn(Float32, size(μ))) .* σ

sampleμσ(μ::T, σ::T) where T = μ .+ oftype(μ, randn(Float32, size(μ))) .* σ

sampleμρ(μ, lv) = sampleμσ(μ, exp.(0.5f0lv))

const samplegauss = sampleμρ
const sampleμlv = sampleμρ

const normal = sampleμσ
reparameterize(μ, σ, sampling=Flux.istraining()) = sampling ? sampleμσ(μ, σ) : μ

# function squeeze(a::AbstractArray{T}, dim=Int) where T
#     s = size(a)
#     reshape(a, (s[i] for i=1:dim-1)..., (s[i] for i=dim+1:ndims(a))...)
# end

# squeeze(a::AbstractArray, f::Base.Callable, dims) = squeeze(f, a, dims)

# squeeze(f::Base.Callable, a::AbstractArray, dims) = @> f(a, dims=dims) squeeze(dims)

# squeeze(a::AbstractArray, dims::Int) where T = squeeze(a, [dims])

# function squeeze(a::AbstractArray{T}, dims::AbstractArray{Int}) where T
#     s = size(a)
#     n = ndims(a)
#     @assert all(s[dims] .== 1)
#     keep_dims = indexin(1:n, dims) .== nothing
#     reshape(a, s[keep_dims])
# end

klqnormal((μ, ρ)) = klqnormal(μ, ρ)
klqnormal(μ::AbstractArray{T}, ρ::AbstractArray{T}) where T = 0.5f0 * sum(@. (exp(ρ) + μ^2 - 1f0 - ρ)) / size(μ)[end]

klqp((μ₁, ρ₁), (μ₂, ρ₂)) = sum(@. 0.5f0(exp(ρ₁ - ρ₂) + (μ₁ - μ₂)^2/exp(ρ₂) - 1 - (ρ₁ - ρ₂))) / size(μ₁)[end]

accuracy(x, y, m) = mean(onecold(cpu(m(x))) .== onecold(cpu(y)))

function accuracy(d, m)
    ŷ, y = @> map(d) do (x, y)
        onecold(cpu(m(x))), onecold(cpu(y))
    end zips cats.(dims=1)
    mean(ŷ .== y)
end

function uniform!(x, a, b)
    rand!(x) .* (b-a) .+ a
end

function copystruct!(a::T, b::U) where {T, U}
    for f in fieldnames(T)
        setfield!(a, f, getfield(b, f))
    end
    b
end

call(f, x) = f(x)


