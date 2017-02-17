import numpy as np
import tensorflow as tf
import time

def train(model, hparams, data):

    # Init
    loss = np.zeros(hparams.steps)
    p_max_c1 = np.zeros(hparams.steps)
    p_max_c2 = np.zeros(hparams.steps)

    error = np.zeros(hparams.steps/hparams.epoch_size)
    er_p_max_c1 = np.zeros(hparams.steps/hparams.epoch_size)
    er_p_max_c2 = np.zeros(hparams.steps/hparams.epoch_size)

    # Get initial data and run through the graph
    a = time.time()
    try:
        for i in range(hparams.steps):
            search_space, template = model.sess.run([data.s_train, data.t_train])

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
                        model.dropout: hparams.dropout}


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
                a = time.time()
                if error[j]==0:
                    raise ValueError('finished')

    except:
        pass
    finally:
        save_path = model.saver.save(model.sess, hparams.model_dir+"model_"+model.id+".ckpt")
        model.train_writer.close()
        model.test_writer.close()
        model.coord.request_stop()
        model.coord.join(model.threads)
        model.sess.close()
        print("Nicely Shutting down")

    return loss

def test(model, hparams, data, iteration):

    sum_dist, sum_p1, sum_p2  = (0, 0, 0)

    steps = len(hparams.pathset)/hparams.batch_size

    with tf.name_scope('Testing'):
        for i in range(steps):
            #s, t = model.sess.run([data.s_test, data.t_test])
            t, s =  data.getTrainBatch()

            model_run =[model.merged,
                        model.l,
                        model.p_max,
                        model.p_max_2]
            feed_dict ={model.image: s, model.template: t, model.dropout: 1}

            summary, ls, p_1, p_2 = model.sess.run(model_run,feed_dict=feed_dict)
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
        t,s = data.getSample(template_shape, source_shape, hparams.resize, metadata, deterministic, pos)
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