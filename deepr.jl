using Flux, JLD2, Statistics
using Flux: DataLoader, Embedding, unsqueeze, maxpool, onehotbatch, onecold

include("utils.jl")

# CONST
max_length        = 100  # maximum length
embed_size        = 32  # embedding size
hidden        = 64  # hidden size
filters        = 20  # number of filter
batchsize        = 30  # batch_size
kernels       = [10,20,30]  # filters' size
epochs    = 5
folds     = 5
pdrop     = 0.5
L1       = 0.0
L2       = 0.0
LR       = 1e-2  # learning rate
SEED     = 1

zips(a::Vector{T}) where T = map(t-> map(x->x[t], a), 1:length(first(a)))

function leftpad(x::AbstractVector, p, t=length(x)+1; truncate=true)
  x = deepcopy(x)
  length(x) < t && (x = vcat(fill(p, t-length(x)), x))
  truncate && (x = x[end-t+1:end])
  x
end

JLD2.@load "./mimic3.pidxyy_unplanvdvp.jld2" pid x y y_unplan vd vp
pid  # patient id

# DATA STRUCTURE
x  # medical records of all patients
x[1]  # medical records of 1st patients: list of diagnoses and procedures
x[1][1]  # list of diagnoses in each episode
x[1][2]  # list of procedures in each episode

# Coding abbreviation
# p·9962: Procedure 9962. Heart countershock NEC
# d·87: Diagnosis Code V87: Contact with and (suspected) exposure to other potentially hazardous chemicals

y  # not use

y_unplan  # to be predicted: Boolean vector of unplanned readmission in 6 months
vd  # list of all diagnosis codes
vp  # list of all procedure codes

xs, ys = x, y_unplan

ds, ps = zips(xs)  # extract diagnoses and procedures
x = ds  # use diagnoses as inputs

# MAKE datasets
join_episode(x) = vcat(split.(x)...)
remove_empty(x) = filter(x->!isempty(x), x)

x2 = @> x join_episode.() remove_empty.() leftpad.("PAD", 100)
x2
x = x2
vx = @> vcat(x...) unique sort  # shorten the vocab if possible
vocab_size = length(vx)  # length of vocab
output_size = 2  # number of classes to predict, binary
PAD = 1
x = @> x indexin.([vx])
x = hcat(x...)
const labels = [0, 1]
y = @> onehotbatch(y_unplan, labels) Float32.() gpu

# DATA SPLIT 4:1
N = size(x)[end] ÷ batchsize * batchsize
x0, y0 = deepcopy(x), deepcopy(y)
fold = 1
it = fold:5:N
ir = setdiff(1:N)
xtrain, ytrain = x0[:, ir], y0[:, ir]
xtest, ytest = x0[:, it], y0[:, it]
trainset = DataLoader((xtrain, ytrain), batchsize=batchsize) |> gpu
testset = DataLoader((xtest, ytest), batchsize=batchsize) |> gpu
x, y = first(trainset)

globalmaxpool(x) = maxpool(x, size(x)[1:2])

model = Chain(
          Embedding(vocab_size, embed_size),
          Dropout(pdrop),
          unsqueeze(3),
          Parallel(vcat, [Chain(
                                Conv((embed_size, w), 1=>filters, relu, pad=0, stride=1),
                                globalmaxpool,
                               ) for w∈kernels]...),
          flatten,
          Dense(length(kernels)*filters, hidden, relu),
          Dense(hidden, output_size),
         ) |> gpu

model[1:4](x) |> size
model[1:5](x) |> size
model(x)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
loss(x, y)

optimizer = ADAM()
ps = params(model)

epoch = 1

function train(epoch)
    epoch
    for (x, y) = trainset
        _, back = Flux.pullback(ps) do
            loss(x, y)
        end
        grad = back(1f0)
        Flux.Optimise.update!(optimizer, ps, grad)
    end
end

function predictall(model, testset)
    ŷ, y = @> map(testset) do (x, y)
        model(x), y
    end zips cats.(dims=2) Array.()
    ŷ, y
end

accuracy(ŷ, y) = mean(onecold(ŷ, labels) .== onecold(y, labels))

function test(testset)
    ŷ, y = predictall(model, testset)
    accuracy(ŷ, y)
end

for epoch = 1:epochs
    train(epoch)
    l = loss(first(trainset)...)
    train_acc = test(trainset)
    test_acc = test(testset)
    @≥ l, train_acc, test_acc round.(digits=3)
    @show epoch, l, train_acc, test_acc
end

