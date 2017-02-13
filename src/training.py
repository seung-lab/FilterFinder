import numpy as np
import tensorflow as tf
import time

def train(model, hparams, data):
    # Set shapes
    source_shape = [hparams.source_width, hparams.source_width]
    template_shape = [hparams.template_width, hparams.template_width]

    # Init
    loss = np.zeros(hparams.steps)
    p_max_c1 = np.zeros(hparams.steps)
    p_max_c2 = np.zeros(hparams.steps)

    error = np.zeros(hparams.steps/hparams.epoch_size)
    er_p_max_c1 = np.zeros(hparams.steps/hparams.epoch_size)
    er_p_max_c2 = np.zeros(hparams.steps/hparams.epoch_size)

    # Get initial data and run through the graph
    a = time.time()

    for i in range(hparams.steps):

        #Check id data is aligned
        if hparams.aligned:
            template, search_space = data.getAlignedSample(template_shape, source_shape, train_set)
        else:
            template, search_space = data.getSample(template_shape, source_shape, hparams.resize, data.metadata)

        #Train step data
        run_metadata = tf.RunMetadata()
        model_run =[model.train_step,
                    model.l,
                    model.p_max,
                    model.p_max_2,
                    model.p,
                    model.kernel_conv[0],
                    model.source_alpha,
                    model.template_alpha,
                    model.merged]

        feed_dict ={model.image: search_space,
                    model.template: template,
                    model.dropout:
                    hparams.dropout}

        #_, ls, p_max_c, p_max_c_2, norm, filt, s_f, t_f, summary
        step = model.sess.run(model_run, feed_dict=feed_dict, run_metadata=run_metadata )

        model.train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        model.train_writer.add_summary(step[-1], i)

        loss[i] = np.absolute(step[1])
        p_max_c1[i] = step[2]
        p_max_c2[i] = step[3]

        #Evaluate
        if i%hparams.epoch_size==0:
            b = time.time()
            j = i/hparams.epoch_size
            error[j], er_p_max_c1[j], er_p_max_c2[j] = test(model,hparams, data, i)
            print ("step: %g, train: %g, test: %g, time %g"%(i,loss[i], error[i/hparams.epoch_size], b-a))
            save_path = model.saver.save(model.sess, hparams.model_dir+"model_"+model.id+".ckpt")
            a = time.time()

    showMultiLoss(loss, p_max_c1, p_max_c2, 100)
    showMultiLoss(error, er_p_max_c1, er_p_max_c2, 20)

    model.train_writer.close()
    model.test_writer.close()

    return loss

def test(model, hparams, data, iteration):

    sum_dist, sum_p1, sum_p2  = (0, 0, 0)

    source_shape = [hparams.source_width, hparams.source_width]
    template_shape = [hparams.template_width, hparams.template_width]

    steps = hparams.eval_batch_size

    pathset = [ (120,9900, 11000), (20, 9900, 11000),
                (60, 16000, 17000),(70, 16000, 17000),
                (400, 8500, 27000),(400, 7000, 27000),
                (300, 7000, 21500),(151, 4500, 5000),
                (51, 18000, 9500), (52, 18000, 7500),
                (55, 18000, 7500), (60, 18100, 8400)]

    if ~hparams.aligned:
        steps = len(pathset)

    with tf.name_scope('Testing'):
        for i in range(steps):
            if hparams.aligned:
                t,s = data.getAlignedSample(template_shape, source_shape, test_set, i)
            else:
                t,s = data.getSample(template_shape, source_shape, hparams.resize, data.metadata, pathset[i][0], pathset[0][1:3])

            summary, ls, p_1, p_2 = model.sess.run([model.merged, model.l, model.p_max, model.p_max_2],
                                                    feed_dict={model.image: s, model.template: t, model.dropout: 1})
            #tf.summary.image('test '+str(i), p, 1)

            sum_dist = sum_dist + np.absolute(p_1-p_2)
            sum_p1 = sum_p1 + p_1
            sum_p2 = sum_p2 + p_2

        model.test_writer.add_summary(summary, iteration)
    return sum_dist/steps, sum_p1/steps, sum_p2/steps

def evaluate(deterministic, source_shape, template_shape, aligned=True, pos=(12334, 4121)):
    sum_dist = 0

    if aligned:
        t,s = data.getAlignedSample(template_shape, source_shape, test_set, deterministic)
    else:
        t,s = data/getSample(template_shape, source_shape, hparams.resize, metadata, deterministic, pos)
    norm, filt, s_f, t_f, P_1, P_2 = sess.run([p, kernel, source_alpha, template_alpha, p_max, p_max_2], feed_dict={image: s, temp: t, drop: 1})
    #print(norm)
    print(s_f)
    #print(s)
    print(P_1-P_2)
    print(P_1, P_2)
    xcsurface(norm)
    show(norm)

    #print(filt)
    #0print((kernel_shape[0]/2,kernel_shape[1]/2))
    #filt[32,32] = 0
    #filt_2[16,16] = 0
    show(filt[:,:])
    #show(filt_2[:,:])
    show(s)
    show(s_f)
    show(t)
    show(t_f)
