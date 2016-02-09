-- Only requirement allowed
require('hdf5')
require('socket') -- allows for accurate runtime measurement

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb, smr, svm', 'classifier to use')
cmd:option('-alpha', '', 'smoothing parameter in naive bayes')
cmd:option('-lambda', '', 'regularization parameter')
cmd:option('-eta', '', 'learning rate')
cmd:option('-mu', '', 'momentum parameter')
cmd:option('-epochs', '', 'number of training epochs')
cmd:option('-minibatch', '', 'number of examples per minibatch')
cmd:option('-output', '', 'file name for output')
cmd:option('-run', 'kfoldcv, valid, testfile, gridsearch, randomsearch', 'experiment to run')
cmd:option('-lmin', '', 'search minimum for lambda')
cmd:option('-lmax', '', 'search maximum for lambda')
cmd:option('-emin', '', 'search minimum for eta')
cmd:option('-emax', '', 'search maximum for eta')
cmd:option('-tries', '', 'number of search points')
cmd:option('-k', '', 'number of splits in kfoldcv')
cmd:option('-kfoldnum', '[0, k]', 'number of iterations to execute in kfoldcv')
cmd:option('-trainfrac','(0, 1)', 'fraction of the training set to use')
cmd:option('-showepochs', '0, 1', 'prints epoch numbers and durations')
cmd:option('-showloss', '0, 1', 'prints loss during training')
cmd:option('-checkgradient', '0, 1', 'checks gradients and prints results')
cmd:option('-momentum', 'mom', 'adds momentum to gradient updates')
-- Parsing
opt = cmd:parse(arg)
print('file:', opt.datafile)
local f = hdf5.open(opt.datafile, 'r')
nclasses = f:read('nclasses'):all():long()[1]
nfeatures = f:read('nfeatures'):all():long()[1]
local train_input = f:read('train_input'):all()
local train_output = f:read('train_output'):all()
local valid_input = nil
local valid_output = nil
local test_input = nil
if opt.datafile == 'SST1.hdf5' or opt.datafile == 'SST2.hdf5' then
    valid_input = f:read('valid_input'):all()
    valid_output = f:read('valid_output'):all()
end
if opt.datafile == 'SST1.hdf5' or opt.datafile == 'SST2.hdf5' or opt.datafile == 'TREC.hdf5' then
    test_input = f:read('test_input'):all()
end

-- Hyperparameters
alpha = tonumber(opt.alpha) or 1 -- smoothing parameter for naive bayes
eta = tonumber(opt.eta) or 1 -- learning rate for softmax regression
lambda = tonumber(opt.lambda) or .5-- regularization parameter for softmax regression
mu = tonumber(opt.mu) or .1
epochs = tonumber(opt.epochs) or 30
mini_batch_size = tonumber(opt.minibatch) or 32
epsilon = 10^(-4) -- gradient checking parameter

lambdamin = tonumber(opt.lmin) or .25
lambdamax = tonumber(opt.lmax) or 2
etamin = tonumber(opt.emin) or .5
etamax = tonumber(opt.emax) or 2.5
tries = tonumber(opt.stries) or 15
kfoldnum = tonumber(opt.kfoldnum) or nil
k = tonumber(k) or 10
trainfrac = tonumber(opt.trainfrac) or 1
showepochs = tonumber(opt.showepochs) or 0
showloss = tonumber(opt.showloss) or 0
checkgradient = tonumber(opt.checkgradient) or 0
momentum = opt.momentum or nil
train_input = train_input:sub(1, math.floor(train_input:size(1)*trainfrac))
train_output = train_output:sub(1, math.floor(train_output:size(1)*trainfrac))

-- Functions
  -- returns a sparse representation of an input vector
function sparserep(x, n)
    out = torch.Tensor(1, n):zero()
    for i = 1, x:size(1) do
        if x[i] ~= 1 then
            out[1][x[i]] = 1
        else
            break
        end
    end
    return out
end

  -- returns a naive bayes hypothesis
function naivebayes(train_input, train_output)
    -- construct count matrix
    local W = torch.zeros(nfeatures, nclasses)
    for i = 1, train_input:size(1) do
        for j = 1, train_input:size(2) do
            if train_input[i][j] == 1 then break end
            W[train_input[i][j]][train_output[i]] = W[train_input[i][j]][train_output[i]] + 1
        end
    end

    -- W doubles as F
    W:add(alpha)
    local b = W:sum(1)
    W:cdiv(b:expand(W:size(1), W:size(2))):log()
    b:log()

    return (function (x) return (x*W):add(b) end)
end

  -- returns a softmax hypothesis with cross-entropy loss
function softmaxreg(train_input, train_output)
    local W = torch.randn(nfeatures, nclasses)
    local b = torch.randn(1, nclasses)
    for i = 1, epochs do
        if showepochs == 1 then print('epoch:', i) end
        starttime = socket.gettime()
        W, b = SGD('softmax', W, b, train_input, train_output, mini_batch_size)
        if showepochs == 1 then print('epoch time:', (socket.gettime() - starttime), 's') end
    end
    return (function (x) return (x*W):add(b) end)
end

-- returns a linear svm hypothesis with hinge loss
function linearsvm(train_input, train_output)
    local W = torch.randn(nfeatures, nclasses)
    local b = torch.randn(1, nclasses)
    for i = 1, epochs do
        if showepochs == 1 then print('epoch:', i) end
        starttime = socket.gettime()
        W, b = SGD('hinge', W, b, train_input, train_output, mini_batch_size)
        if showepochs == 1 then print('epoch time:', (socket.gettime() - starttime), 's') end
    end
    return (function (x) return (x*W):add(b) end)
end

-- shuffles data and oversees gradient descent
function SGD(alg, W, b, input, output, mini_batch_size)
    -- shuffle data
    p = torch.randperm(input:size(1)):long()
    input = input:index(1, p)
    output = output:index(1, p)

    vW = torch.zeros(W:size())
    vb = torch.zeros(b:size())
    -- step through the data
    step = math.floor((input:size(1) + 1)/mini_batch_size)
    for i = 1, input:size(1)-step, step do
        W, b, vW, vb = gradient_update(W, b, input:sub(i, i+step-1), output:sub(i, i+step-1), alg, vW, vb)
    end
    return W, b
end

-- gradient updates for softmax and svm
function gradient_update(W, b, input, output, alg, vW, vb)
    local z = torch.Tensor(b:size())
    local yhat = torch.Tensor(z:size())
    local ym = torch.Tensor(z:size()) -- temp storage
    local lgradz = torch.Tensor(z:size())
    local gradW = torch.zeros(W:size())
    local gradb = torch.zeros(b:size())
    local x = torch.Tensor(nfeatures)
    local loss = 0
    for i = 1, input:size(1) do
        x = sparserep(input[i], nfeatures)
        z:mm(x, W):add(b)
        max, argmax = z:max(2)
        m = max[1][1]; argmax = argmax[1][1]
        if alg == 'softmax' then
            -- log-sum-exp with computational trick
            yhat:add(z, -m)
            logpart =yhat:exp():sum(2):log():add(m)[1][1]
            z:add(-logpart):exp()
            loss = loss - math.log(z[1][output[i]])
            j = 0
            lgradz:zero()
            lgradz:apply(function () j = j+1 return (j == output[i] and z[1][j]-1 or z[1][j]) end)
        elseif alg == 'hinge' then
            -- find cprime
            ym[1][output[i]] = ym[1][(argmax > 1 and argmax - 1 or argmax + 1)] - 1
            max, cprime = ym:max(2); cprime = cprime[1][1]
            j = 0
            lgradz:zero()
            lgradz:apply(function () j=j+1;
                return (z[1][output[i]] - z[1][cprime] > 1 and 0 or j==cprime and 1 or j == output[i] and -1 or 0) end)
            loss = loss + math.max(0, 1 - (z[1][output[i]] - z[1][cprime]))
        end
        -- optional gradient checking
        if checkgradient == 1 then
            gw = torch.zeros(gradW:size())
            gw:addr(x:squeeze(), lgradz:squeeze())
            grad_check(gw, lgradz, x, W, b, output[i], alg == 'softmax' and softmaxloss or alg == 'hinge' and hingeloss)
        end
        gradW:addr(x:squeeze(), lgradz:squeeze())
        gradb:add(lgradz)
    end
    if showloss == 1 then print('loss:', loss) end

    -- update
    if momentum == 'mom' then
        vW:mul(mu):add(-eta/input:size(1), gradW)
        vb:mul(mu):add(-eta/input:size(1), gradb)
    end
    W:add( - eta*lambda/input:size(1), W)
    b:add( - eta*lambda/input:size(1), b)
    if not momentum then
        W:add(-eta/input:size(1), gradW)
        b:add(-eta/input:size(1), gradb)
    elseif momentum == 'mom' then
        W:add(vW)
        b:add(vb)
    end
    return W, b, vW, vb
end


-- loss of softmax reg
function softmaxloss(x, W, b, ans)
    local z = torch.Tensor(b:size())
    local yhat = torch.Tensor(b:size())
    z:mm(x, W):add(b)
    max, argmax = z:max(2); m = max[1][1]
    logpart = torch.add(z, -m):exp():sum(2):log():add(m)[1][1]
    z:add(-logpart):exp()
    return -math.log(z[1][ans])
end

-- multiclass hinge loss
function hingeloss(x, W, b, ans)
    local z = torch.Tensor(b:size())
    local ym = torch.Tensor(b:size())
    z:mm(x, W):add(b)
    ym:copy(z)
    max, argmax = ym:max(2); argmax = argmax[1][1]
    ym[1][ans] = ym[1][argmax] - 1
    max, cprime = ym:max(2); cprime = cprime[1][1]
    return math.max(0, 1 - (z[1][ans] - z[1][cprime]))
end

-- finite difference gradient checking
function grad_check(gradW, gradb, x, W, b, ans, loss, wb)
    wb = wb or 'wb'
    local Wp = torch.Tensor(W:size())
    local Wm = torch.Tensor(W:size())
    local bp = torch.Tensor(b:size())
    local bm = torch.Tensor(b:size())
    local z = torch.Tensor(b:size())
    print('checking')
    for i = 1, W:size(1) do
        for j = 1, W:size(2) do
            if wb == 'w' or wb == 'wb' then
                Wp:copy(W)
                Wm:copy(W)
                Wp[i][j] = W[i][j] + epsilon
                Wm[i][j] = W[i][j] - epsilon
                lp = loss(x, Wp, b, ans)
                lm = loss(x, Wm, b, ans)
                g = (lp - lm)/(2*epsilon)
                max = math.max(math.abs(g), math.abs(gradW[i][j]))
                rerr = math.abs(g - gradW[i][j])/max
                if max ~= 0 then
                    print('W:', 'i:', i, 'j:', j, 'relative error:', rerr, 'analytic g:', g, 'gradw i j', gradW[i][j])
                end
            end
            if i == 1 and (wb == 'b' or wb == 'wb') then
                bp:copy(b)
                bm:copy(b)
                bp[1][j] = b[1][j] + epsilon
                bm[1][j] = b[1][j] - epsilon
                lp = loss(x, W, bp, ans)
                lm = loss(x, W, bm, ans)
                g = (lp - lm)/(2*epsilon)
                max = math.max(math.abs(g), math.abs(gradb[1][j]))
                rerr = math.abs(g - gradb[1][j])/max

                if max ~= 0 then
                    print('b:', 'i:', i, 'j:', j, 'relative error:', rerr, 'analytic g:', g, 'gradb i j', gradb[1][j])
                end
                if rerr > 10^(-10) then
                    -- print(x)
                    z:mm(x, W):add(b)
                    print(z, ans, cprime)
                end
            end
        end
    end
end

-- tests a hypothesis h on input and output sets
function test(h, input, output)
    correct = 0
    for i = 1, input:size(1) do
        maxval, pred = h(sparserep(input[i], nfeatures)):max(2)
        correct = correct + (pred[1][1] == output[i] and 1 or 0) -- pred is stored in a 1x1 longtensor
        -- print(i .. ':', output[i], pred)
    end
    return correct/input:size(1)
end

  -- performs k-fold cross validation on algorithm alg (performs /num/ of /k/ possible validations)
function kfoldcv(alg, input, output, k, num)
    num = num or k
    p = torch.randperm(input:size(1)):long()
    input = input:index(1, p)
    output = output:index(1, p)

    test_size = math.floor(input:size(1)/k)
    accuracies = torch.zeros(k)
    for i = 1, num do
        -- train on batches 1,...,i-1, i+1,...,k and test on batch i
        trainind = torch.LongTensor(input:size(1) - test_size)
        testind = torch.LongTensor(test_size)
        j = 0
        trainind:apply(function() j = j + (j == test_size*(i-1) and test_size+1 or 1) return j end)
        j = (i-1)*test_size
        testind:apply(function() j = j+1 return j end)
        -- print('train:', trainind)
        -- print('test:', testind)
        h = alg(input:index(1, trainind), output:index(1, trainind))
        accuracies[i] = test(h, input:index(1, testind), output:index(1, testind))
        print(accuracies[i])
    end
    return accuracies:sum(1)/num
end

-- outputs test hypotheses in kaggle format
function csv(h, input)
    f = io.open(opt.output, 'w')
    f:write('ID,Category\n')
    for i = 1, input:size(1) do
        maxval, pred = h(sparserep(input[i], nfeatures)):max(2); pred = pred[1][1]
        f:write(i .. ','.. pred .. '\n')
    end
    f:close()
end


-- execution
if opt.classifier == 'nb' then
    alg = naivebayes
elseif opt.classifier == 'smr' then
    alg = softmaxreg
elseif opt.classifier == 'svm' then
    alg = linearsvm
end

if opt.run == 'kfoldcv' then
    print(kfoldcv(alg, train_input, train_output, k, kfoldnum))
elseif opt.run == 'valid' then
    if valid_input then
        h = alg(train_input, train_output)
        print('validation set performance:', test(h, valid_input, valid_output))
    else
        print('no validation set')
    end
elseif opt.run == 'testfile' then
    h = alg(train_input, train_output)
    csv(h, test_input)
elseif opt.run == 'randsearch' then
    math.randomseed(os.time())

    tries = 15
    results = torch.zeros(tries, 3)
    for i = 1, tries do
        lambda = math.random()*(lambdamax - lambdamin) + lambdamin
        eta = math.random()*(etamax - etamin) + etamin
        print('lambda:', lambda, 'eta:', eta)
        results[i][1] = lambda; results[i][2] = eta
        if valid_input then
            h = alg(train_input, train_output)
            results[i][3] = test(h, valid_input, valid_output)
        else
            results[i][3] = kfoldcv(alg, train_input, train_output, 10, kfoldnum)
        end
        print(results[i][3])
    end
    max, argmax = results:max(1)
    print('best: eta =', results[argmax[1][3]][1], 'lambda =', results[argmax[1][3]][2], 'accuracy:', max[1][1])
end
