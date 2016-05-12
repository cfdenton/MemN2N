require("hdf5")
require("nn")
require("rnn")
require("dpnn")
require("nngraph")
require("optim")
-- penlight library which allows pretty table printing for debug
tp = require("pl.pretty")

cmd = torch.CmdLine()
cmd:option('-dataset', 'babi', 'babi or mctest')
cmd:option('-datafile', '', 'data file')
cmd:option('-model', 'memn2n', 'prediction model')
cmd:option('-optim', 'sgd', 'optimization algorithm')
cmd:option('-ngram', 2, 'ngram size for ngram model')
cmd:option('-epochs', 100, 'training epochs')
cmd:option('-pre_epochs', 30, 'pretraining epochs')
cmd:option('-batch', 32, 'batch size')
cmd:option('-eta', .01, 'learning rate')
cmd:option('-pre_eta', .001, 'learning rate for pretraining')
cmd:option('-decay', .5, 'learning rate decay')
cmd:option('-emb', 20, 'nn word embedding size')
cmd:option('-nhops', 3, 'number of hops for the memn2n')
cmd:option('-dropout', 0, 'dropout')
cmd:option('-debug', false, '')
cmd:option('-renorm', 40, 'gradient normalization')
cmd:option('-pos_enc', false, 'use position encoding')
cmd:option('-temp_enc', true, 'use temporal encoding')
cmd:option('-share', 'adj', 'type of weight-tying to use')
cmd:option('-memsize', 50, 'size of memory')
cmd:option('-anneal_schedule', 25, 'epochs between anneals')
cmd:option('-ls', false, 'Use linear start')
cmd:option('-rn', 0, 'Prepend random memories')
cmd:option('-no_print', false, 'dont print epochs')
cmd:option('-heat', 0, 'learned (-1) or fixed (> 1) temperature parameter')
cmd:option('-sentence_encoding', 'bow', 'type of sentence encoding')
cmd:option('-pre_train', false, 'do pretraining')
cmd:option('-linear_transform', false, 'linear after rnn')
cmd:option('-show_attn', false, 'print attention dist')
cmd:option('-match', 'dot', 'method for computing match with memories [dot, cosine, linear]')
cmd:option('-conv_size', 2, 'convolution window size')
cmd:option('-use_glove', false, 'use glove word2vec embeddings')

function memn2n_model(pre_training)
    -- context and query taken in forward (with time and pos_weights as applicable)
    local context = nn.Identity()()
    local query = nn.Identity()()
    local time = nn.Identity()()
    local pos_weights
    if opt.pos_enc then
        pos_weights = nn.Identity()()
    end
    local query_emb, enc
    -- encode query
    if opt.sentence_encoding == 'bow' then
        query_emb = nn.LookupTable(V, opt.emb)(query)
        query_enc = nn.Sum(1, 2)(nn.Squeeze()(query_emb))
    elseif opt.sentence_encoding == 'lstm' or
      opt.sentence_encoding == 'gru' then
        q_lookup = nn.LookupTable(V, opt.emb)(query)
        q_sq = nn.Squeeze()(q_lookup)
        q_split = nn.SplitTable(1, 2)(q_sq)
        if opt.sentence_encoding == 'lstm' then
            q_rnn = nn.LSTM(opt.emb, opt.emb)
        elseif opt.sentence_encoding == 'gru' then
            q_rnn = nn.GRU(opt.emb, opt.emb)
        end
        q_seq = nn.Sequencer(q_rnn)(q_split)
        q_view = nn.Sequencer(nn.View(-1, 1, opt.emb))(q_seq)
        q_join = nn.JoinTable(1, 2)(q_view)
        q_ave = nn.Mean(1, 2)(q_join)
        if opt.linear_transform then
            query_emb = nn.Linear(opt.emb, opt.emb, false)(q_ave)
        else
            query_emb = nn.Identity()(q_ave)
        end
        query_enc = nn.Identity()(query_emb)

        if opt.pre_train then
            q_lookup.data.module.weight:copy(pre_training[1])
            local q_rnn_param, _ = q_rnn:getParameters()
            q_rnn_param:copy(pre_training[2])
        end
    elseif opt.sentence_encoding == 'conv' then
        query_emb = nn.LookupTable(V, opt.emb)(query)
        q_sq = nn.Squeeze()(query_emb)
        q_conv = nn.TemporalConvolution(opt.emb, opt.emb, opt.conv_size)(q_sq)
        q_pool = nn.Mean(1, 2)(q_conv)
        query_enc = nn.Identity()(q_pool)
    end
    local query_in = nn.Squeeze()(query_enc)

    local hops = {}
    local memory_embs = {}
    local time_embs = {}
    if opt.sentence_encoding == 'conv' then conv_layers = {} end
    hops[0] = query_in

    local match
    if opt.match == 'linear' then
        match = nn.Linear(2*opt.emb, 1, false)
    end

    for i = 1, opt.nhops do
        local hop_query = nn.View(-1, 1)(hops[i-1])
        -- encode memories for matching
        local A, A_pw, A_out, A_enc, A_lookup, A_sq, A_split, A_rnn, A_select
        if opt.sentence_encoding == 'bow' then
            A = nn.LookupTable(V, opt.emb)(context)
            if opt.pos_enc then
                A_pw = nn.CMulTable()({A, pos_weights})
            else
                A_pw = nn.Identity()(A)
            end
            A_out = nn.Sum(1, 2)(A_pw)
        elseif opt.sentence_encoding == 'lstm'
          or opt.sentence_encoding == 'gru' then
            A_lookup = nn.LookupTable(V, opt.emb)(context)
            A_sq = nn.Squeeze()(A_lookup)
            A_split = nn.SplitTable(1, 2)(A_sq)
            if opt.sentence_encoding == 'lstm' then
                A_rnn = nn.LSTM(opt.emb, opt.emb)
            elseif opt.sentence_encoding == 'gru' then
                A_rnn = nn.GRU(opt.emb, opt.emb)
            end
            A_seq = nn.Sequencer(A_rnn)(A_split)
            A_view = nn.Sequencer(nn.View(-1, 1, opt.emb))(A_seq)
            A_join = nn.JoinTable(1, 2)(A_view)
            A_ave = nn.Mean(1, 2)(A_join)
            if opt.linear_transform then
                A = nn.Linear(opt.emb, opt.emb, false)(A_ave)
            else
                A = nn.Identity()(A_ave)
            end
            A_out = nn.Identity()(A)
            if opt.pre_train then
                A_lookup.data.module.weight:copy(pre_training[1])
                local A_rnn_param, _ = A_rnn:getParameters()
                A_rnn_param:copy(pre_training[2])
            end
            A_lookup.data.module:share(q_lookup.data.module, 'weight', 'gradWeight')
            A_rnn:share(q_rnn, 'weight', 'gradWeight')
        elseif opt.sentence_encoding == 'conv' then
            A = nn.LookupTable(V, opt.emb)(context)
            A_conv = nn.TemporalConvolution(opt.emb, opt.emb, opt.conv_size)(A)
            A_pool = nn.Mean(1, 2)(A_conv)
            A_out = nn.Identity()(A_pool)
        end
        local mem_in
        -- add temporal encoding
        local T_A = nn.LookupTable(opt.memsize, opt.emb)(time)
        if opt.temp_enc then
            mem_in = nn.CAddTable()({A_out, T_A})
        else
            mem_in = nn.Identity()(A_out)
        end
        -- create temperature
        local heat
        if opt.heat == -1 then
            heat = nn.Mul()
        elseif opt.heat >= 1 then
            heat = nn.MulConstant(opt.heat)
        else
            heat = nn.Identity()
        end
        -- compute attention
        local p1, mem_in_norm, hop_query_norm
        if opt.match == 'dot' then
            p1 = nn.MM(false, false)({mem_in, hop_query})
        elseif opt.match == 'cosine' then
            mem_in_norm = nn.Normalize(2)(mem_in)
            hop_query_norm = nn.Normalize(2)(hop_query)
            p1 = nn.MM(false, false)({mem_in_norm, hop_query_norm})
        elseif opt.match == 'linear' then
            replicate = nn.Replicate(1)(nn.Squeeze()(hop_query))
            print_rep = nn.Identity()(replicate)
            print_mem = nn.Identity()(mem_in)
            concat_p = nn.JoinTable(2, 2)({print_mem, print_rep})
            p1 = nn.Linear(2*opt.emb, 1)(concat_p)
            p1.data.module:share(match, 'weight', 'gradWeight')
        end
        local heater = heat(p1)
        local p = nn.View(-1,1)(nn.SoftMax()(nn.Squeeze(2)(heater)))
        local C, C_pw, C_out, C_enc, C_lookup, C_sq, C_split, C_rnn, C_select
        -- output encoding
        if opt.sentence_encoding == 'bow' then
            C = nn.LookupTable(V, opt.emb)(context)
            if opt.pos_enc then
                C_pw = nn.CMulTable()({C, pos_weights})
            else
                C_pw = nn.Identity()(C)
            end
            C_out = nn.Sum(1, 2)(C_pw)
        elseif opt.sentence_encoding == 'lstm' or
          opt.sentence_encoding == 'gru' then
            C_lookup = nn.LookupTable(V, opt.emb)(context)
            C_sq = nn.Squeeze()(C_lookup)
            C_split = nn.SplitTable(1, 2)(C_sq)
            if opt.sentence_encoding == 'lstm' then
                C_rnn = nn.LSTM(opt.emb, opt.emb)
            elseif opt.sentence_encoding == 'gru' then
                C_rnn = nn.GRU(opt.emb, opt.emb)
            end
            C_seq = nn.Sequencer(C_rnn)(C_split)
            C_view = nn.Sequencer(nn.View(-1, 1, opt.emb))(C_seq)
            C_join = nn.JoinTable(1, 2)(C_view)
            C_ave = nn.Mean(1, 2)(C_join)
            if opt.linear_transform then
                C = nn.Linear(opt.emb, opt.emb, false)(C_ave)
            else
                C = nn.Identity()(C_ave)
            end
            C_out = nn.Identity()(C)
            if opt.pretrain then
                C_lookup.data.module.weight:copy(pre_training[1])
                local C_rnn_param, _ = C_rnn:getParameters()
                C_rnn_param:copy(pre_training[2])
            end
            C_lookup.data.module:share(q_lookup.data.module, 'weight', 'gradWeight')
            C_rnn:share(q_rnn, 'weight', 'gradWeight')
        elseif opt.sentence_encoding == 'conv' then
            C = nn.LookupTable(V, opt.emb)(context)
            C_conv = nn.TemporalConvolution(opt.emb, opt.emb, opt.conv_size)(C)
            C_pool = nn.Mean(1, 2)(C_conv)
            C_out = nn.Identity()(C_pool)
        end
        -- temporal encoding
        local T_C = nn.LookupTable(opt.memsize, opt.emb)(time)
        local mem_out
        if opt.temp_enc then
            mem_out = nn.CAddTable()({C_out, T_C})
        else
            mem_out = nn.Identity()(C_out)
        end
        -- combine memory and hop input
        local o = nn.Squeeze()(nn.MM(true, false)({mem_out, p}))
        local u = nn.Squeeze()(hop_query)
        local add = nn.CAddTable()({u, o})
        local hop_out
        if opt.share == 'rnn' then
            hop_out = nn.Dropout(opt.dropout(nn.Linear(opt.emb, opt.emb, false)(add)))
        else
            hop_out = nn.Dropout(opt.dropout)(add)
        end
        -- store nodes for sharing
        table.insert(hops, hop_out)
        table.insert(memory_embs, {A, C})
        if opt.temp_enc then
            table.insert(time_embs, {T_A, T_C})
        end
        if opt.sentence_encoding == 'conv' then
            table.insert(conv_layers, {A_conv, C_conv})
        end
    end

    -- final output
    local lin = nn.Linear(opt.emb, V, false)(hops[opt.nhops])
    local out = nn.View(1, -1)(lin)

    -- share parameters
    if opt.share == 'adj' then
        local B = query_emb.data.module
        local A1 = memory_embs[1][1].data.module
        B:share(A1, 'weight', 'bias', 'gradWeight', 'gradBias')
        for i = 1, opt.nhops-1 do
            local C = memory_embs[i][2].data.module
            local A = memory_embs[i+1][1].data.module
            A:share(C, 'weight', 'bias', 'gradWeight', 'gradBias')
            if opt.temp_enc then
                local T_C = time_embs[i][2].data.module
                local T_A = time_embs[i+1][1].data.module
                T_A:share(T_C, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
            if opt.sentence_encoding == 'conv' then
                local C_conv = conv_layers[i][2].data.module
                local A_conv = conv_layers[i+1][1].data.module
                A_conv:share(C_conv, 'weight', 'gradWeight')
            end
        end
        if opt.sentence_encoding == 'bow' or opt.sentence_encoding == 'conv' then
            W = lin.data.module
            Ck = memory_embs[opt.nhops][2].data.module
            W:share(Ck, 'weight', 'gradWeight')
        end
    elseif opt.share == 'rnn' then
        for i = 1, opt.nhops-1 do
            local A_prev = memory_embs[i][1].data.module
            local C_prev = memory_embs[i][2].data.module
            local A_next = memory_embs[i+1][1].data.module
            local C_next = memory_embs[i+1][2].data.module
            A_next:share(A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            C_next:share(C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            if opt.temp_enc then
                local T_A_prev = time_embs[i][1].data.module
                local T_C_prev = time_embs[i][2].data.module
                local T_A_next = time_embs[i+1][1].data.module
                local T_C_next = time_embs[i+1][2].data.module
                T_A_next:share(T_A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
                T_C_next:share(T_C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
            if opt.sentence_encoding == 'conv' then
                local A_conv_prev = conv_layers[i][1].data.module
                local C_conv_prev = conv_layers[i][2].data.module
                local A_conv_next = conv_layers[i+1][1].data.module
                local C_conv_next = conv_layers[i+1][2].data.module
                A_conv_next:share(A_conv_prev, 'weight', 'gradWeight')
                C_conv_next:share(C_conv_prev, 'weight', 'gradWeight')
            end
        end
    end

    -- wrap model
    local model
    if opt.temp_enc and opt.pos_enc then
        model = nn.gModule({context, query, pos_weights, time}, {out})
    elseif opt.pos_enc then
        model = nn.gModule({context, query, pos_weights}, {out})
    elseif opt.temp_enc then
        model = nn.gModule({context, query, time}, {out})
    else
        model = nn.gModule({context, query}, {out})
    end
    -- zero gradParameters to fix nasty bug
    local parameters, gradParameters = model:getParameters()
    gradParameters:zero()
    local criterion = nn.CrossEntropyCriterion()
    return model, criterion
end

function memn2n_mc_model_simple()
    -- context and query taken in forward
    -- input nodes, embed query
    local context = nn.Identity()()
    local query = nn.Identity()()
    local choices = nn.Identity()()
    local time = nn.Identity()()
    local pos_weights = nn.Identity()()
    local query_emb = nn.LookupTable(V, opt.emb)(query)
    local query_in = nn.Sum(1, 2)(nn.Squeeze(2)(query_emb))

    local hops = {}
    local memory_embs = {}
    local choice_embs = {}
    local time_embs = {}
    hops[0] = query_in

    for i = 1, opt.nhops do
        local hop_query = nn.View(-1, 1)(hops[i-1])
        local A = nn.LookupTable(V, opt.emb)(context)
        local A_pw = nn.Sum(1, 2)(nn.CMulTable()({A, pos_weights}))
        local mem_in
        local T_A = nn.LookupTable(opt.memsize, opt.emb)(time)
        if opt.temp_enc then
            mem_in = nn.CAddTable()({A_pw, T_A})
        else
            mem_in = nn.Identity()(A_pw)
        end
        local heat
        if opt.heat == -1 then
            heat = nn.Mul()
        elseif opt.heat >= 1 then
            heat = nn.MulConstant(opt.heat)
        else
            heat = nn.Identity()
        end
        local mem_p1 = heat(nn.MM(false, false)({mem_in, hop_query}))
        local mem_p = nn.View(-1,1)(nn.SoftMax()(nn.Squeeze(2)(mem_p1)))
        local C = nn.LookupTable(V, opt.emb)(context)
        local C_pw = nn.Sum(1, 2)(nn.CMulTable()({C, pos_weights}))
        local T_C = nn.LookupTable(opt.memsize, opt.emb)(time)
        local mem_out
        if opt.temp_enc then
            prc = nn.Identity()(C_pw)
            prt = nn.Identity()(T_C)
            mem_out = nn.CAddTable()({prc, prt})
        else
            mem_out = nn.Identity()(C_pw)
        end
        local o = nn.Squeeze()(nn.MM(true, false)({mem_out, mem_p}))
        local u = nn.Squeeze()(hop_query)
        local add = nn.CAddTable()({u, o})
        local hop_out = nn.Dropout(opt.dropout)(add)
        table.insert(hops, hop_out)
        table.insert(memory_embs, {A, C})
        if opt.temp_enc then
            table.insert(time_embs, {T_A, T_C})
        end
    end

    -- final output
    local choice_emb = nn.LookupTable(V, opt.emb)(choices)
    local choice_in = nn.Sum(1,2)(choice_emb)
    local lin = nn.MM(false, false)({choice_in, nn.View(-1,1)(hops[opt.nhops])})
    local out = nn.View(1, -1)(lin)

    -- share parameters
    if opt.share == 'adj' then
        local B = query_emb.data.module
        local A1 = memory_embs[1][1].data.module
        B:share(A1, 'weight', 'bias', 'gradWeight', 'gradBias')
        for i = 1, opt.nhops-1 do
            local C = memory_embs[i][2].data.module
            local A = memory_embs[i+1][1].data.module
            A:share(C, 'weight', 'bias', 'gradWeight', 'gradBias')
            if opt.temp_enc then
                local T_C = time_embs[i][2].data.module
                local T_A = time_embs[i+1][1].data.module
                T_A:share(T_C, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif opt.share == 'rnn' then
        for i = 1, opt.nhops-1 do
            local A_prev = memory_embs[i][1].data.module
            local C_prev = memory_embs[i][2].data.module
            local A_next = memory_embs[i+1][1].data.module
            local C_next = memory_embs[i+1][2].data.module
            A_next:share(A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            C_next:share(C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')

            if opt.temp_enc then
                local T_A_prev = time_embs[i][1].data.module
                local T_C_prev = time_embs[i][2].data.module
                local T_A_next = time_embs[i+1][1].data.module
                local T_C_next = time_embs[i+1][2].data.module
                T_A_next:share(T_A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
                T_C_next:share(T_C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    end

    local model
    if opt.temp_enc then
        model = nn.gModule({context, query, choices, pos_weights, time}, {out})
    else
        model = nn.gModule({context, query, choices, pos_weights}, {out})
    end
    local criterion = nn.CrossEntropyCriterion()
    return model, criterion
end


function memn2n_mc_model()
    local context = nn.Identity()()
    local query = nn.Identity()()
    local choices = nn.Identity()()
    local time = nn.Identity()()
    local pos_weights = nn.Identity()()
    local query_emb = nn.LookupTable(V, opt.emb)(query)
    local query_in = nn.Sum(1, 2)(nn.Squeeze(2)(query_emb))

    local hops = {}
    local memory_embs = {}
    local choice_embs = {}
    local time_embs = {}
    hops[0] = query_in

    for i = 1, opt.nhops do
        local hop_query = nn.View(-1, 1)(hops[i-1])
        local A = nn.LookupTable(V, opt.emb)(context)
        local A_pw = nn.Sum(1, 2)(nn.CMulTable()({A, pos_weights}))
        local mem_in
        local T_A = nn.LookupTable(opt.memsize, opt.emb)(time)
        if opt.temp_enc then
            mem_in = nn.CAddTable()({A_pw, T_A})
        else
            mem_in = nn.Identity()(A_pw)
        end
        local heat
        if opt.heat == -1 then
            heat = nn.Mul()
        elseif opt.heat >= 1 then
            heat = nn.MulConstant(opt.heat)
        else
            heat = nn.Identity()
        end
        local mem_p1 = heat(nn.MM(false, false)({mem_in, hop_query}))
        local mem_p = nn.View(-1,1)(nn.SoftMax()(nn.Squeeze(2)(mem_p1)))
        local C = nn.LookupTable(V, opt.emb)(context)
        local C_pw = nn.Sum(1, 2)(nn.CMulTable()({C, pos_weights}))
        local T_C = nn.LookupTable(opt.memsize, opt.emb)(time)
        local mem_out
        if opt.temp_enc then
            prc = nn.Identity()(C_pw)
            prt = nn.Identity()(T_C)
            mem_out = nn.CAddTable()({prc, prt})
        else
            mem_out = nn.Identity()(C_pw)
        end
        local o = nn.Squeeze()(nn.MM(true, false)({mem_out, mem_p}))
        local u = nn.Squeeze()(hop_query)
        local add = nn.CAddTable()({u, o})
        local choice_in_emb = nn.LookupTable(V, opt.emb)(choices)
        local choice_in = nn.Sum(1,2)(choice_in_emb)
        local choice_p1 = nn.MM(false, false)({choice_in, nn.View(-1,1)(add)})
        local choice_p  = nn.View(-1,1)(nn.SoftMax()(nn.Squeeze(2)(choice_p1)))
        local choice_out_emb = nn.LookupTable(V, opt.emb)(choices)
        local choice_out = nn.Sum(1,2)(choice_out_emb)
        local choice_o = nn.Squeeze()(nn.MM(true, false)({choice_out, choice_p}))
        local add2 = nn.CAddTable()({add, choice_o})
        local hop_out = nn.Dropout(opt.dropout)(add2)
        table.insert(hops, hop_out)
        table.insert(memory_embs, {A, C})
        table.insert(choice_embs, {choice_in_emb, choice_out_emb})
        if opt.temp_enc then
            table.insert(time_embs, {T_A, T_C})
        end
    end

    -- final output
    local lin = nn.Linear(opt.emb, 4, false)(hops[opt.nhops])
    local out = nn.View(1, -1)(lin)

    -- share parameters
    if opt.share == 'adj' then
        local B = query_emb.data.module
        local A1 = memory_embs[1][1].data.module
        B:share(A1, 'weight', 'bias', 'gradWeight', 'gradBias')
        for i = 1, opt.nhops-1 do
            local C = memory_embs[i][2].data.module
            local A = memory_embs[i+1][1].data.module
            A:share(C, 'weight', 'bias', 'gradWeight', 'gradBias')
            local choice_out_prev = choice_embs[i][2].data.module
            local choice_in_next = choice_embs[i+1][1].data.module
            choice_in_next:share(choice_out_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            if opt.temp_enc then
                local T_C = time_embs[i][2].data.module
                local T_A = time_embs[i+1][1].data.module
                T_A:share(T_C, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif opt.share == 'rnn' then
        for i = 1, opt.nhops-1 do
            local A_prev = memory_embs[i][1].data.module
            local C_prev = memory_embs[i][2].data.module
            local A_next = memory_embs[i+1][1].data.module
            local C_next = memory_embs[i+1][2].data.module
            A_next:share(A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            C_next:share(C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')

            if opt.temp_enc then
                local T_A_prev = time_embs[i][1].data.module
                local T_C_prev = time_embs[i][2].data.module
                local T_A_next = time_embs[i+1][1].data.module
                local T_C_next = time_embs[i+1][2].data.module
                T_A_next:share(T_A_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
                T_C_next:share(T_C_prev, 'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    end

    local model
    if opt.temp_enc then
        model = nn.gModule({context, query, choices, pos_weights, time}, {out})
    else
        model = nn.gModule({context, query, choices, pos_weights}, {out})
    end
    local criterion = nn.CrossEntropyCriterion()
    return model, criterion
end



function pos_encoding(J)
    -- Create position weighting for sentence of length J
    local d = opt.emb
    local pos_weights = torch.zeros(opt.memsize, J, d)
    for i = 1, opt.memsize do
        for j = 1, J do
            for k = 1, d  do
                pos_weights[i][j][k] = (1 - (j/J)) - (k/d)*(1 - ((2*j)/J))
            end
        end
    end
    return pos_weights
end

-- define ngram model (logistic regression)
function ngram_model()
    local model = nn.Sequential()
    model:add(nn.LookupTable(ngram_features, V))
    model:add(nn.Sum(1))
    model:add(nn.SoftMax())
    local criterion = nn.CrossEntropyCriterion()
    return model, criterion
end

-- ngram tensor to string rep e.g. 1-2-1-5-4
function t2s(tens)
    local s = ''
    for i = 1, tens:size(1)-1 do s = s .. tens[i] .. '-' end
    return s .. tens[tens:size(1)]
end

-- construct ngram dictionary
function ngramdict()
    local idx = 1
    local idxs = torch.LongTensor(opt.ngram):fill(1)
    local pos = 1
    local dict = {}
    while true do
        dict[t2s(idxs)] = idx
        if idxs:eq(V):all() then break end
        idx = idx + 1
        for i = 1, opt.ngram do
            if idxs[i] == V then
                idxs[i] = 1
            else
                idxs[i] = idxs[i] + 1
                break
            end
        end
    end
    return dict
end

-- bag of ngrams from sentences sharing at least one word with query
function ngramfeat(context, query)
    -- identify which sentences to choose
    local snum = 0
    local idxs = {}
    for i = 1, context:size(1) do
      for j = 1, query:size(1) do
        if context[i]:index(1, context[i]:nonzero():squeeze()):eq(query[j]):any() then
            snum = snum + 1
            idxs[snum] = i
            break
        end
      end
    end
    -- determine how many ngrams these comprise; pad by opt.ngram-1
    local relevant = context:index(1, torch.LongTensor(idxs))
    local num = relevant:ne(0):long():sum(2)
    num = num:add(opt.ngram-1)
    -- create bag and fill with ngrams
    local bag = torch.LongTensor(num:sum(1):squeeze())
    local idx = 1
    local ngram = torch.LongTensor(opt.ngram)
    for i = 1, relevant:size(1) do
        ngram:fill(BUFFER)
        for j = 1, num[i][1] do
            if opt.ngram > 1 then
                ngram:sub(1, opt.ngram-1):copy(ngram:sub(2, opt.ngram))
            end
            if j <= relevant[i]:size(1) and relevant[i][j] ~= 0 then
                ngram[opt.ngram] = relevant[i][j]
            else
                ngram[opt.ngram] = BUFFER
            end
            bag[idx] = ngramdict[t2s(ngram)]
            idx = idx + 1
        end
    end
    return bag
end

function train(model, criterion, words, query_idx, idxs, ans_words,
    valid_words, valid_query_idx, valid_idxs, valid_ans_words, pos_weights, noinit)

    parameters, gradParameters = model:getParameters()
    -- initialize parameters from N(0,.01)
    if not noinit then parameters:copy(torch.randn(parameters:size()):mul(.1)) end
    local state
    local model_out_name = opt.model .. "-"
    local fname_idx, _ = string.find(opt.datafile, "%.")
    model_out_name = model_out_name .. string.sub(opt.datafile, 1, fname_idx-1) .. ".t7"

    for epoch = 1, opt.epochs do
        model:training()
        state = {
              learningRate = opt.eta
        }
        local loss, ex_loss, ex = 0, 0, 0
        local startstory, endstory, query_off = 1, 1, 0
        local story
        local context = torch.Tensor(opt.memsize, words:size(2))
        for i = 1, query_idx:size(1) do
            -- zero out padding weights
            if opt.model == 'memn2n' and opt.sentence_encoding ~= 'lstm' and
              opt.model ~= 'gru' then
                local lt_list = model:findModules('nn.LookupTable')
                for j = 1, #lt_list do
                    if lt_list[j].weight:size(1) == V then
                        lt_list[j].weight[1]:zero()
                    end
                end
                if opt.heat == -1 then
                    local heat_list = model:findModules('nn.Mul')
                    local heat = torch.Tensor(#heat_list)
                    for j = 1, #heat_list do
                        if heat_list[j].weight[1] < 1 then
                            heat_list[j].weight:copy(torch.ones(1))
                        end
                        heat[j] = 1/heat_list[j].weight[1]
                    end
                    if i == 3 then
                        print("Heat:", heat)
                    end
                end
            end
            -- get story for query i
            if idxs[endstory + 2] == 1 then
                query_off = endstory + 1
                startstory = endstory + 2
                story = nil
            end
            endstory = query_off + query_idx[i] - 1
            if story == nil then
                story = words:sub(startstory, endstory)
            elseif startstory <= endstory then
                story = story:cat(words:sub(startstory, endstory), 1)
            end

            if opt.model == 'memn2n' then
                local softmax_list = model:findModules('nn.SoftMax')
                if #softmax_list > 0 and i == 3  then
                    attention = torch.zeros(#softmax_list, softmax_list[1].output:size(1))
                    for j = 1, #softmax_list do
                        attention[j]:copy(softmax_list[j].output)
                    end
                    if opt.show_attn then print("Attention:", attention) end
                end
            end

            -- construct input
            if opt.model == 'ngram' then
                input = ngramfeat(story, words[query_off + query_idx[i]])
            elseif opt.model == 'memn2n' then
                -- construct fixed context window of max size opt.memsize from story
                if story:size(1) < opt.memsize then
                    for i = story:size(1), 1, -1 do
                        context[i]:copy(story[story:size(1)-i+1])
                    end
                    input = {context:sub(1, story:size(1)), words[query_off + query_idx[i]]:view(-1, 1)}
                else
                    for i = opt.memsize, 1, -1 do
                        context[i]:copy(story[story:size(1)-i+1])
                    end
                    input = {context, words[query_off + query_idx[i]]:view(-1, 1)}
                end
                if opt.pos_enc then
                    if story:size(1) < opt.memsize then
                        table.insert(input, pos_weights:sub(1, story:size(1)))
                    else
                        table.insert(input, pos_weights)
                    end
                end
                if opt.temp_enc then
                    if story:size(1) < opt.memsize then
                        table.insert(input, torch.range(1, story:size(1)))
                    else
                        table.insert(input, torch.range(1, opt.memsize))
                    end
                end
            end

            -- adjust sizes for linear matching
            for _, m in ipairs(model.modules) do
                if torch.type(m) == 'nn.Replicate' then
                    m.nfeatures = input[1]:size(1)
                end
            end
            -- forward/backward, accumulate gradparameters
            local pred = model:forward(input)
            ex_loss = criterion:forward(pred, torch.LongTensor({ans_words[i]}))
            local dloss_dout = criterion:backward(pred, torch.LongTensor({ans_words[i]}))
            model:backward(input, dloss_dout)

            -- update at each batch
            if i % opt.batch == 0 then
                if opt.renorm ~= 0 then
                    gradParameters:view(-1, 1):renorm(2, 2, opt.renorm)
                end
                local losseval = function (x)
                    collectgarbage()
                    if x ~= parameters then
                        parameters:copy(x)
                    end
                    return loss, gradParameters
                end
                if opt.optim == 'sgd' then
                    parameters, _ = optim.sgd(losseval, parameters, state)
                elseif opt.optim == 'adagrad' then
                    parameters, _ = optim.adagrad(losseval, parameters, state)
                elseif opt.optim == 'rmsprop' then
                    parameters, _ = optim.rmsprop(losseval, parameters, state)
                end
                model:zeroGradParameters()
            end
            loss = loss + ex_loss
            startstory = query_off + query_idx[i] + 1
        end
        torch.save(model_out_name, model)

        -- print progress
        local train_acc = eval(model, words, query_idx, idxs, ans_words, pos_weights)
        local valid_acc = eval(model, valid_words, valid_query_idx, valid_idxs, valid_ans_words, pos_weights)
        if not opt.no_print then print("epoch", epoch, "loss", loss/query_idx:size(1),
               "train", train_acc, "valid", valid_acc) end

        -- anneal learning rate
        if epoch % opt.anneal_schedule == 0 then
            opt.eta = opt.eta * opt.decay
            if not opt.no_print then print("eta changed:", opt.eta) end
        end
    end
    return model
end

function train_mctest(model, criterion, words, query_idx, idxs, ans_words, choice_words, valid_words, valid_query_idx, valid_idxs, valid_ans_words, valid_choice_words, pos_weights, noinit)
    parameters, gradParameters = model:getParameters()
    -- initialize parameters from N(0,.01)
    if noinit then
        if opt.use_glove then
            -- initialize lookup tables to word vecs
            local lt_list = model:findModules('nn.LookupTable')
            for i = 1, #lt_list do
                if lt_list[i].weight:size(1) == word_vecs:size(1) then
                    lt_list[i].weight:copy(word_vecs:mul(.1))
                end
            end
        end
    else
        parameters:copy(torch.randn(parameters:size()):mul(.1))
        if opt.use_glove then
            -- initialize lookup tables to word vecs
            local lt_list = model:findModules('nn.LookupTable')
            for i = 1, #lt_list do
                if lt_list[i].weight:size(1) == word_vecs:size(1) then
                    lt_list[i].weight:copy(word_vecs:mul(.1))
                end
            end
        end
    end
    local state
    local model_out_name = opt.model .. "-"
    local fname_idx, _ = string.find(opt.datafile, "%.")
    model_out_name = model_out_name .. string.sub(opt.datafile, 1, fname_idx-1) .. ".t7"

    for epoch = 1, opt.epochs do
        model:training()
        if opt.optim == 'sgd' or opt.optim == 'adagrad' then
            state = {
              learningRate = opt.eta
            }
        end
        local loss, ex_loss, ex = 0, 0, 0
        local startstory, endstory, query_off = 1, 1, 0
        local story
        local context = torch.Tensor(opt.memsize, words:size(2))
        --model:training()
        for i = 1, query_idx:size(1) do
            -- zero out padding weights
            local lt_list = model:findModules('nn.LookupTable')
            for j = 1, #lt_list do
                if lt_list[j].weight:size(1) == V then
                    lt_list[j].weight[1]:zero()
                end
            end
            if opt.heat == -1 then
                local heat_list = model:findModules('nn.Mul')
                local heat = torch.Tensor(#heat_list)
                for j = 1, #heat_list do
                    if heat_list[j].weight[1] < 1 then
                        heat_list[j].weight:copy(torch.ones(1))
                    end
                    heat[j] = 1/heat_list[j].weight[1]
                end
                if i == 3 then
                    print("Heat:", heat)
                end
            end

            -- get story for query i
            if idxs[endstory + 2] == 1 then
                query_off = endstory + 1
                startstory = endstory + 2
                story = nil
            end
            endstory = query_off + query_idx[i] - 1
            if story == nil then
                story = words:sub(startstory, endstory)
            elseif startstory <= endstory then
                story = story:cat(words:sub(startstory, endstory), 1)
            end

            -- construct fixed context window of size opt.memsize from story
            if story:size(1) < opt.memsize then
                for i = story:size(1), 1, -1 do
                    context[i]:copy(story[story:size(1)-i+1])
                end
                input = {context:sub(1, story:size(1)), words[query_off + query_idx[i]]:view(-1, 1), choice_words[i], pos_weights:sub(1, story:size(1))}
            else
                for i = opt.memsize, 1, -1 do
                    context[i]:copy(story[story:size(1)-i+1])
                end
                input = {context, words[query_off + query_idx[i]]:view(-1, 1), choice_words[i], pos_weights}

            end
            if opt.temp_enc then
                if story:size(1) < opt.memsize then
                    input[5] = torch.range(1, story:size(1))
                else
                    input[5] = torch.range(1, opt.memsize)
                end
            end

            local pred = model:forward(input)
            ex_loss = criterion:forward(pred:squeeze(), ans_words[i])
            local dloss_dout = criterion:backward(pred:squeeze(), ans_words[i])
            model:backward(input, dloss_dout)

            if i % opt.batch == 0 then
                if opt.renorm ~= 0 then
                    gradParameters:view(-1, 1):renorm(2, 2, opt.renorm)
                end
                local losseval = function (x)
                    collectgarbage()
                    if x ~= parameters then
                        parameters:copy(x)
                    end
                    return loss, gradParameters
                end
                if opt.optim == 'sgd' then
                    parameters, _ = optim.sgd(losseval, parameters, state)
                elseif opt.optim == 'adagrad' then
                    parameters, _ = optim.adagrad(losseval, parameters, state)
                end
                model:zeroGradParameters()
            end
            loss = loss + ex_loss
            startstory = query_off + query_idx[i] + 1
        end
        torch.save(model_out_name, model)
        local train_acc = eval_mctest(model, words, query_idx, idxs, choice_words, ans_words, pos_weights)
        local valid_acc = eval_mctest(model, valid_words, valid_query_idx, valid_idxs, valid_choice_words, valid_ans_words, pos_weights)
        -- local total_acc = (train_acc * 280 + valid_acc * 120) / 400
        if not opt.no_print then print("epoch", epoch, "loss", loss/query_idx:size(1),
               "train", train_acc, "valid", valid_acc) end
        -- annealing learning rate
        if epoch % opt.anneal_schedule == 0 then
            opt.eta = opt.eta * opt.decay
            if not opt.no_print then print("eta changed:", opt.eta) end
        end
    end
    return model
end

function eval(model, words, query_idx, idxs, ans_words, pos_weights)
    model:evaluate()
    local startstory, endstory, query_off = 1, 1, 0
    local correct = 0
    local story
    local context = torch.Tensor(opt.memsize, words:size(2))
    for i = 1, query_idx:size(1) do
        collectgarbage()
        -- construct story for query i
        if idxs[endstory + 2] == 1 then
            query_off = endstory + 1
            startstory = endstory + 2
            story = nil
        end
        endstory = query_off + query_idx[i] - 1
        if story == nil then
            story = words:sub(startstory, endstory)
        elseif startstory <= endstory then
            story = story:cat(words:sub(startstory, endstory), 1)
        end

        if opt.model == 'ngram' then
            input = ngramfeat(story, words[query_off + query_idx[i]])
        elseif opt.model == 'memn2n' then
            -- construct fixed context window of size opt.memsize from story
            if story:size(1) < opt.memsize then
                for i = story:size(1), 1, -1 do
                    context[i]:copy(story[story:size(1)-i+1])
                end
                input = {context:sub(1, story:size(1)), words[query_off + query_idx[i]]:view(-1, 1)}
            else
                for i = opt.memsize, 1, -1 do
                    context[i]:copy(story[story:size(1)-i+1])
                end
                input = {context, words[query_off + query_idx[i]]:view(-1, 1)}
            end
            if opt.pos_enc then
                if story:size(1) < opt.memsize then
                    table.insert(input, pos_weights:sub(1, story:size(1)))
                else
                    table.insert(input, pos_weights)
                end
            end
            if opt.temp_enc then
                if story:size(1) < opt.memsize then
                    table.insert(input, torch.range(1, story:size(1)))
                else
                    table.insert(input, torch.range(1, opt.memsize))
                end
            end
        end
        for _, m in ipairs(model.modules) do
            if torch.type(m) == 'nn.Replicate' then
                m.nfeatures = input[1]:size(1)
            end
        end
        pred = model:forward(input)
        if opt.model == 'ngram' then
            _, pred = pred:max(1)
        else
            _, pred = pred:max(2)
        end

        if pred:squeeze() == ans_words[i] then correct = correct + 1 end
        startstory = query_off + query_idx[i] + 1
    end
    return correct/query_idx:size(1)
end


function eval_mctest(model, words, query_idx, idxs, choice_words, ans_words, pos_weights)
    model:evaluate()
    local startstory, endstory, query_off = 1, 1, 0
    local correct = 0
    local story
    local context = torch.Tensor(opt.memsize, words:size(2))
    for i = 1, query_idx:size(1) do
        collectgarbage()
        -- construct story for query i
        if idxs[endstory + 2] == 1 then
            query_off = endstory + 1
            startstory = endstory + 2
            story = nil
        end
        endstory = query_off + query_idx[i] - 1
        if story == nil then
            story = words:sub(startstory, endstory)
        elseif startstory <= endstory then
            story = story:cat(words:sub(startstory, endstory), 1)
        end

        -- construct fixed context window of size opt.memsize from story
        if story:size(1) < opt.memsize then
            for i = story:size(1), 1, -1 do
                context[i]:copy(story[story:size(1)-i+1])
            end
            input = {context:sub(1, story:size(1)), words[query_off + query_idx[i]]:view(-1, 1), choice_words[i], pos_weights:sub(1, story:size(1))}
        else
            for i = opt.memsize, 1, -1 do
                context[i]:copy(story[story:size(1)-i+1])
            end
            input = {context, words[query_off + query_idx[i]]:view(-1, 1), choice_words[i], pos_weights}
        end
        if opt.temp_enc then
            if story:size(1) < opt.memsize then
                input[5] = torch.range(1, story:size(1))
            else
                input[5] = torch.range(1, opt.memsize)
            end
        end

        pred = model:forward(input)
        if i > 10 and i < 21 and words == valid_words then
            print(nn.SoftMax():forward(pred))
        end
        _, pred = pred:max(2)

        if pred:squeeze() == ans_words[i] then correct = correct + 1 end
        if i > 10 and i < 21 and words == valid_words then
            print(pred:squeeze(), ans_words[i], i)
        end
        startstory = query_off + query_idx[i] + 1
    end
    return correct/query_idx:size(1)
end

function train_lm(words, valid_words)
    -- define model
    local pre_model = nn.Sequential()
    local lookup = nn.LookupTable(V, opt.emb)
    pre_model:add(lookup)
    pre_model:add(nn.SplitTable(1, 2))
    local rnn
    if opt.sentence_encoding == 'lstm' then
        rnn = nn.LSTM(opt.emb, opt.emb)
    elseif opt.sentence_encoding == 'gru' then
        rnn = nn.GRU(opt.emb, opt.emb)
    end
    local seq = nn.Sequential()
    seq:add(rnn)
    seq:add(nn.Linear(opt.emb, V))
    pre_model:add(nn.Sequencer(seq))
    pre_model:add(nn.NarrowTable(1, words:size(2)-1))
    local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())

    local output = words:narrow(2, 2, words:size(2)-1)
    local parameters, gradParameters = pre_model:getParameters()
    local state = {learningRate = opt.pre_eta}

    print("epoch", 0, "loss", "n/a", "perp", eval_lm(pre_model, valid_words))
    for epoch = 1, opt.pre_epochs do
        local loss = 0
        for batch_start = 1, words:size(1), opt.batch do
            local batch_end = math.min(words:size(1), batch_start + opt.batch - 1)
            local batch_in = words:sub(batch_start, batch_end)
            local batch_out = output:sub(batch_start, batch_end):split(1, 2)

            local func = function (x)
                if x ~= parameters then
                    collectgarbage()
                    parameters:copy(x)
                end
                local pred = pre_model:forward(batch_in)
                local loss = criterion:forward(pred, batch_out)
                local dloss_dout = criterion:backward(pred, batch_out)
                pre_model:backward(batch_in, dloss_dout)
                return loss, gradParameters
            end

            if opt.optim == 'sgd' then
                parameters, batch_loss = optim.sgd(func, parameters, state)
            elseif opt.optim == 'adagrad' then
                parameters, batch_loss = optim.adagrad(func, parameters, state)
            elseif opt.optim == 'rmsprop' then
                parameters, batch_loss = optim.rmsprop(func, parameters, state)
            end
            loss = loss + batch_loss[1]

        end
        print("epoch", epoch, "loss", loss, "perp", eval_lm(pre_model, valid_words))
    end

    local lookup_param, _ = lookup:getParameters()
    local rnn_param, _ = rnn:getParameters()
    return {lookup_param, rnn_param}
end

function eval_lm(model, words)
    local output_words = words:narrow(2, 2, valid_words:size(2)-1):contiguous():view(-1)
    local pred_words = model:forward(valid_words)
    pred_words = nn.Sequencer(nn.View(-1, 1, V)):forward(pred_words)
    pred_words = nn.JoinTable(1, 2):forward(pred_words):view(-1, V)
    pred_words = nn.SoftMax():forward(pred_words)
    local probs = torch.Tensor(output_words:nElement())
    for i = 1, probs:size(1) do
        probs[i] = pred_words[i][output_words[i]]
    end
    local perp = probs:log():div(probs:size(1)):mul(-1):sum(1):exp()
    return perp[1]
end


function train_lm(words, valid_words)
    -- define model
    local pre_model = nn.Sequential()
    local lookup = nn.LookupTable(V, opt.emb)
    pre_model:add(lookup)
    pre_model:add(nn.SplitTable(1, 2))
    local lstm = nn.LSTM(opt.emb, opt.emb)
    local seq = nn.Sequential()
    seq:add(lstm)
    seq:add(nn.Linear(opt.emb, V))
    pre_model:add(nn.Sequencer(seq))
    pre_model:add(nn.NarrowTable(1, words:size(2)-1))
    local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())

    local output = words:narrow(2, 2, words:size(2)-1)
    local parameters, gradParameters = pre_model:getParameters()
    local state = {learningRate = opt.pre_eta}

    print("epoch", 0, "loss", "n/a", "perp", eval_lm(pre_model, valid_words))
    for epoch = 1, opt.pre_epochs do
        local loss = 0
        for batch_start = 1, words:size(1), opt.batch do
            local batch_end = math.min(words:size(1), batch_start + opt.batch - 1)
            local batch_in = words:sub(batch_start, batch_end)
            local batch_out = output:sub(batch_start, batch_end):split(1, 2)

            local func = function (x)
                if x ~= parameters then
                    collectgarbage()
                    parameters:copy(x)
                end
                local pred = pre_model:forward(batch_in)
                local loss = criterion:forward(pred, batch_out)
                local dloss_dout = criterion:backward(pred, batch_out)
                pre_model:backward(batch_in, dloss_dout)
                return loss, gradParameters
            end

            if opt.optim == 'sgd' then
                parameters, batch_loss = optim.sgd(func, parameters, state)
            elseif opt.optim == 'adagrad' then
                parameters, batch_loss = optim.adagrad(func, parameters, state)
            elseif opt.optim == 'rmsprop' then
                parameters, batch_loss = optim.rmsprop(func, parameters, state)
            end
            loss = loss + batch_loss[1]

        end
        print("epoch", epoch, "loss", loss, "perp", eval_lm(pre_model, valid_words))
    end

    local lookup_param, _ = lookup:getParameters()
    local lstm_param, _ = lstm:getParameters()
    return {lookup_param, lstm_param}
end

function eval_lm(model, words)
    local output_words = words:narrow(2, 2, valid_words:size(2)-1):contiguous():view(-1)
    local pred_words = model:forward(valid_words)
    pred_words = nn.Sequencer(nn.View(-1, 1, V)):forward(pred_words)
    pred_words = nn.JoinTable(1, 2):forward(pred_words):view(-1, V)
    pred_words = nn.SoftMax():forward(pred_words)
    local probs = torch.Tensor(output_words:nElement())
    for i = 1, probs:size(1) do
        probs[i] = pred_words[i][output_words[i]]
    end
    local perp = probs:log():div(probs:size(1)):mul(-1):sum(1):exp()
    return perp[1]
end

function main()
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    train_words = f:read('train_words'):all():long()
    train_idxs = f:read('train_idxs'):all():long()
    train_query_idx = f:read('train_query_idx'):all():long()
    train_ans_words = f:read('train_ans_words'):all():long()
    valid_words = f:read('valid_words'):all():long()
    valid_idxs = f:read('valid_idxs'):all():long()
    valid_query_idx = f:read('valid_query_idx'):all():long()
    valid_ans_words = f:read('valid_ans_words'):all():long()

    if opt.dataset == 'mctest' then
        train_choice_words = f:read('train_choice_words'):all():long()
        valid_choice_words = f:read('valid_choice_words'):all():long()
        test_choice_words = f:read('test_choice_words'):all():long()
        test_words = f:read('test_words'):all():long()
        test_idxs = f:read('test_idxs'):all():long()
        test_query_idx = f:read('test_query_idx'):all():long()
        test_ans_words = f:read('test_ans_words'):all():long()
        if opt.use_glove == true then
            word_vecs = f:read('word_vecs'):all():double()
            opt.emb = 50
        end
    end
    V = f:read('vocabsize'):all():long()[1]
    BUFFER = V

    -- pos weights
    local pos_weights
    if opt.pos_enc then
        pos_weights = pos_encoding(train_words:size(2))
    else
        pos_weights = torch.ones(opt.memsize, train_words:size(2), opt.emb)
    end

	local attention
    tp.dump(opt)
    if opt.pre_train then
        print("Pre-training:")
        pre_training = train_lm(train_words, valid_words)
    end
    local model, criterion
    if opt.model == 'ngram' then
        ngram_features = V^opt.ngram
        ngramdict = ngramdict()
        model, criterion = ngram_model()
    elseif opt.model == 'memn2n' and opt.dataset == 'babi' then
        if opt.pre_train then print("Regular training:") end
        model, criterion = memn2n_model(pre_training)
        if opt.ls then
            if not opt.no_print then print("Performing linear start") end
            local model_ls = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
            model_ls:replace(function(module)
               if torch.typename(module) == 'nn.SoftMax' then
                  return nn.Identity()
               else
                  return module
               end
            end)
            local eta = opt.eta
            local epochs = opt.epochs
            opt.eta = .005
            opt.epochs = 10
            train(model_ls, criterion, train_words, train_query_idx, train_idxs,
            train_ans_words, valid_words, valid_query_idx, valid_idxs, valid_ans_words, pos_weights)
            opt.eta = eta
            opt.epochs = epochs
            if not opt.no_print then print("Back to regularly scheduled training") end
        end
        model = train(model, criterion, train_words, train_query_idx, train_idxs, train_ans_words, valid_words, valid_query_idx, valid_idxs, valid_ans_words, pos_weights, opt.ls)
        print("Train accuracy:", eval(model, train_words, train_query_idx, train_idxs, train_ans_words, pos_weights))
        print("Valid accuracy:", eval(model, valid_words, valid_query_idx, valid_idxs, valid_ans_words, pos_weights))
        print("Test accuracy:")
    elseif opt.model == 'memn2n' and opt.dataset == 'mctest' then
        model, criterion = memn2n_mc_model_simple()
        if opt.ls then
            if not opt.no_print then print("Performing linear start") end
            local model_ls = model:clone('weight', 'bias', 'gradWeight', 'gradBias')
            model_ls:replace(function(module)
               if torch.typename(module) == 'nn.SoftMax' then
                  return nn.Identity()
               else
                  return module
               end
            end)
            local eta = opt.eta
            local epochs = opt.epochs
            opt.eta = .005
            opt.epochs = 10
            train_mctest(model_ls, criterion, train_words, train_query_idx,
            train_idxs, train_ans_words, train_choice_words, valid_words, valid_query_idx, valid_idxs, valid_ans_words, valid_choice_words, pos_weights)
            opt.eta = eta
            opt.epochs = epochs
            if not opt.no_print then print("Back to regularly scheduled training") end
        end

        model = train_mctest(model, criterion, train_words, train_query_idx, train_idxs,
        train_ans_words, train_choice_words, valid_words, valid_query_idx, valid_idxs, valid_ans_words, valid_choice_words, pos_weights, opt.ls)
        print("Train accuracy:", eval_mctest(model, train_words, train_query_idx, train_idxs, train_choice_words, train_ans_words, pos_weights))
        print("Valid accuracy:", eval_mctest(model, valid_words, valid_query_idx, valid_idxs, valid_choice_words, valid_ans_words, pos_weights))
        print("Test accuracy:", eval_mctest(model, test_words, test_query_idx, test_idxs, test_choice_words, test_ans_words, pos_weights))
    end
    if opt.pre_train then print("Regular training:") end
end


main()
