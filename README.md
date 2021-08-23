## Example code for Deepr

### Example input data

```
JLD2.@load "./mimic3.pidxyy_unplanvdvp.jld2" pid x y y_unplan vd vp
```

### DATA STRUCTURE

`x`:  medical records of all patients

`x[1]`:  medical records of 1st patients: list of diagnoses and procedures

`x[1][1]`:  list of diagnoses in each episode

`x[1][2]`:  list of procedures in each episode

`pid`:  patient id

### Coding abbreviation

`p·9962`: Procedure 9962. Heart countershock NEC

`d·87`: Diagnosis Code V87: Contact with and (suspected) exposure to other potentially hazardous chemicals

`y`:  not use

`y_unplan`:  to be predicted: Boolean vector of unplanned readmission in 6 months

`vd`:  list of all diagnosis codes

`vp`:  list of all procedure codes

### Example Model

```
julia> model = Chain(
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
```

### Training

```

julia> for epoch = 1:EPOCH
           train(epoch)
           l = loss(first(trainset)...)
           train_acc = test(trainset)
           test_acc = test(testset)
           @≥ l, train_acc, test_acc round.(digits=3)
           @show epoch, l, train_acc, test_acc
       end
(epoch, l, train_acc, test_acc) = (1, 0.603f0, 0.704, 0.69)
(epoch, l, train_acc, test_acc) = (2, 0.625f0, 0.695, 0.69)
(epoch, l, train_acc, test_acc) = (3, 0.581f0, 0.712, 0.715)
(epoch, l, train_acc, test_acc) = (4, 0.537f0, 0.749, 0.751)
(epoch, l, train_acc, test_acc) = (5, 0.466f0, 0.841, 0.852)

```

Note: This example dataset is for demonstration only. It may have problems.

