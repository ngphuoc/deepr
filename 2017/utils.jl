using Knet
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
using Lazy
using MacroTools: @forward
using AutoGrad: Value

const KA{T,N}=KnetArray{T,N}
const AR{T,N} = Union{Array{T,N}, Value{Array{T,N}}}
const KR{T,N} = Union{KnetArray{T,N}, Value{KnetArray{T,N}}}
const AKR{T,N} = Union{AR{T,N}, KR{T,N}}

import Base.getindex

struct Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.length, Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!
@forward Chain.layers Base.iterate

(c::Chain)(x) = foldl((x, m) -> m(x), c.layers; init = x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

struct Embed{T}
  W::T
end

Embed(in::Integer, out::Integer) = Embed(param(out, in))

(a::Embed)(x) = a.W[:,x]

struct Dense{F,S,T}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity)
  return Dense(param(out, in), param0(out), σ)
end

function (a::Dense)(x)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

struct Conv{F,A,V}
  σ::F
  weight::A
  bias::V
  stride::Int
  pad::Int
  dilation::Int
end

Conv(w, b, σ = identity; stride = 1, pad = 0, dilation = 1) where {T,N} =
  Conv(σ, w, b, stride, pad, dilation)

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; stride = 1, pad = 0, dilation = 1) where N =
  Conv(param(k..., ch...), param0(ch[2]), σ, stride = stride, pad = pad, dilation = dilation)

function (c::Conv)(x)
  z = conv4(c.weight, x; stride=c.stride, padding=c.pad, upscale=c.dilation) .+ reshape(c.bias, (1,1,:,1))
  c.σ == identity ? z : c.σ.(z)
end

pool1(x) = Knet.pool(x; window=size(x)[1:2])

maxpool(x::T, k::S; pad = 0, stride = first(k)) where {T,S} =
  Knet.pool(x, padding=pad, stride=stride, mode=0)

(fs::Array)(x...;kw...) = map(f->f(x...; kw...), fs)

squeezeall(a::T) where T = reshape(a, (Base.Iterators.filter(x -> x!=1, size(a))...,))


function getindex(x::KnetArray{T,3}, i::Colon, j, k) where T
  I,J,K = size(x)
  ix = to_indices(x, (i,j,k))
  I′,J′,K′ = length.(ix)
  js,ks = ix[2:3]
  typeof(js)<:Int && (js=[js])
  typeof(ks)<:Int && (ks=[ks])
  js = repeat(js, outer=K′);
  ks = repeat(ks, inner=J′);
  ix = sub2ind((J,K), js, ks);
  x = reshape(x,(I,J*K));
  @> x[:,ix] reshape((I′,J′,K′))
end

function getindex(x::KR{T,4}, ::Colon, J::Int64, ::Colon, ::Colon) where T
  i,j,k,l = size(x)
  @> x reshape(i,j,k*l) getindex(:, j, :) reshape(i,k,l)
end

function setindex!(x::KR{T,4}, y, ::Colon, J::Int64, ::Colon, ::Colon) where T
  i,j,k,l = size(x)
  @> x reshape(i,j,k*l) setindex!(y, :, j, :) reshape(i,k,l)
end

getindex(x::KnetArray, ::Colon, ::Colon, ::Colon) = x # fix ambiguity with method in rnn.jl

function setindex!(x::KnetArray{T,3}, y, c::Colon, j, k) where T
  I,J,K = size(x)
  ix = to_indices(x, (c,j,k))
  I′,J′,K′ = length.(ix)
  js,ks = ix[2:3]
  typeof(js)<:Int && (js=[js])
  typeof(ks)<:Int && (ks=[ks])
  js = repeat(js, outer=K′);
  ks = repeat(ks, inner=J′);
  ix = sub2ind((J,K), js, ks);
  #= @show ix size(x) size(y) size(ix) =#
  x = reshape(x,(I,J*K));
  if length(y) == I*length(ix)
    y = reshape(y,(I,:))
  end
  #= @show @which setindex!(x, y, :, ix) =#
  #= setindex!(x, y, :, ix) =#
  Knet.unsafe_setindex!(x,KnetArray{T}(y),c,KnetArray{Int32}(ix))
  #= println("done setindex") =#
end

using Knet: Index3

function getindex(A::KnetArray{T,3}, ::Colon, ::Colon, I::Index3) where T
    B = reshape(A, stride(A,3), size(A,3))
    reshape(B[:,I], size(A,1), size(A,2))
end

function setindex!(x::KnetArray{T,3}, y, ::Colon, ::Colon, I::Index3) where T
    reshape(x, stride(x,3), size(x,3))[:,I] = y
    return x
end

