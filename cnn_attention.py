def ConvolutionalLSTM():
      question = self.question
      answer = self.get_answer()

      # add embedding layers
      weights = np.load(self.config['initial_embed_weights'])
      embedding = Embedding(input_dim=self.config['n_words'],
                            output_dim=weights.shape[1],
                            weights=[weights])
      question_embedding = embedding(question)
      answer_embedding = embedding(answer)

      f_rnn = LSTM(141, return_sequences=True, consume_less='mem')
      b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)

      qf_rnn = f_rnn(question_embedding)
      qb_rnn = b_rnn(question_embedding)
      question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)

      af_rnn = f_rnn(answer_embedding)
      ab_rnn = b_rnn(answer_embedding)
      answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)

      # cnn
      cnns = [Convolution1D(filter_length=filter_length,
                        nb_filter=500,
                        activation='tanh',
                        border_mode='same') for filter_length in [1, 2, 3, 5]]
      question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
      answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

      maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
      maxpool.supports_masking = True
      question_pool = maxpool(question_cnn)
      answer_pool = maxpool(answer_cnn)

      return question_pool, answer_pool


def build_model(opts, verbose=False):
    k = 2 * opts.lstm_units  # 300
    L = opts.xmaxlen  # 20
    N = opts.xmaxlen + opts.ymaxlen + 1  # for delim
    print "x len", L, "total len", N
    print "k", k, "L", L

    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    x = Embedding(output_dim=opts.emb, input_dim=opts.max_features, input_length=N, name='x')(main_input)
    drop_out = Dropout(0.1, name='dropout')(x)
    lstm_fwd = LSTM(opts.lstm_units, return_sequences=True, name='lstm_fwd')(drop_out)
    lstm_bwd = LSTM(opts.lstm_units, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
    bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
    drop_out = Dropout(0.1)(bilstm)
    h_n = Lambda(get_H_n, output_shape=(k,), name="h_n")(drop_out)
    Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
    WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name="merged", mode='sum')
    M = Activation('tanh', name="M")(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)

    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged)
    out = Dense(3, activation='softmax')(h_star)
    output = out
    model = Model(input=[main_input], output=output)
    if verbose:
        model.summary()
    # plot(model, 'model.png')
    # # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    # model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(options.lr))
    return model


def new_attention():

      input_dim = 32
      hidden = 32

      #The LSTM  model -  output_shape = (batch, step, hidden)
      model1 = Sequential()
      model1.add(LSTM(input_dim=input_dim, output_dim=hidden, input_length=step, return_sequences=True))

      #The weight model  - actual output shape  = (batch, step)
      # after reshape : output_shape = (batch, step,  hidden)
      model2 = Sequential()
      model2.add(Dense(input_dim=input_dim, output_dim=step))
      model2.add(Activation('softmax')) # Learn a probability distribution over each  step.
      #Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
      model2.add(RepeatVector(hidden))
      model2.add(Permute(2, 1))

      #The final model which gives the weighted sum:
      model = Sequential()
      model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
      model.add(TimeDistributedMerge('sum')) # Sum the weighted elements.

      model.compile(loss='mse', optimizer='sgd')
