# -*- coding: utf-8 -*-
"""
Integrate a model with the Deep Hybrid DataCloud API.
"""

import json
import argparse
import pkg_resources
# import project's deep_config.py
import src.deep_config as cfg
from aiohttp.web import HTTPBadRequest

## Authorization
from flaat import Flaat
flaat = Flaat()


def _catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def _fields_to_dict(fields_in):
    """
    Example function to convert mashmallow fields to dict()
    """
    dict_out = {}
    
    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, 
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split('.', 1)
    module[0] = "conex-generator"

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR,
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    ### One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    ### One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name': None,
        'version': None,
        'summary': None,
        'home-page': None,
        'author': None,
        'author-email': None,
        'license': None,
        'help-train' : train_args,
        'help-predict' : predict_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # e.g. prepare the data


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


@_catch_error
def predict(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    if (not any([kwargs['urls'], kwargs['files']]) or
            all([kwargs['urls'], kwargs['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if kwargs['files']:
        kwargs['files'] = [kwargs['files']]  # patch until list is available
        return _predict_data(kwargs)
    elif kwargs['urls']:
        kwargs['urls'] = [kwargs['urls']]  # patch until list is available
        return _predict_url(kwargs)


def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    message = 'Not implemented (predict_data())'
    message = {"Error": message}
    return message


def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


###
# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
###
@flaat.login_required() # Allows only authorized people to train
def train(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    message = { "status": "ok",
                "training": [],
              }

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)

    # 1. implement your training here

    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import src
    log = src.utils.getLogger(__name__, level="warning")

    import tensorflow as tf
    import numpy as np
    import timeit
    import typing

    # script input
    run = train_args["dataset"]
    save_prefix = train_args["save_name"]
    cache_path = os.path.join(train_args["cache_path"], run)
    epochs = train_args["epochs"]
    batchsize = train_args["batchsize"]

    # input processing
    savepath = os.path.join(src.models.get_path(), "deephdc", save_prefix)
    logpath = os.path.join(savepath, "log")

    # get data
    data = src.data.processed.load_data(run)
    pd : tf.data.Dataset = data[0]
    ed : tf.data.Dataset = data[1]
    label : tf.data.Dataset = data[2]
    metadata : typing.Dict = data[3]

    # parse metadata
    numdata = metadata["length"]

    pd_maxdata = np.array(metadata["particle_distribution"]["max_data"],
                          dtype=np.float32)
    ed_maxdata = np.array(metadata["energy_deposit"]["max_data"],
                          dtype=np.float32)

    pd_depth = metadata["particle_distribution"]["depth"]
    pd_depthlen = len(pd_depth)
    ed_depth = metadata["energy_deposit"]["depth"]
    ed_depthlen = len(ed_depth)
    assert pd_depthlen == ed_depthlen
    depthlen = pd_depthlen

    pd_mindepthlen = metadata["particle_distribution"]["min_depthlen"]
    pd_mindepth = pd_depth[pd_mindepthlen-1]
    ed_mindepthlen = metadata["energy_deposit"]["min_depthlen"]
    ed_mindepth = ed_depth[ed_mindepthlen-1]
    assert pd_mindepthlen == ed_mindepthlen
    mindepthlen = pd_mindepthlen

    # prepare data
    def cast_to_float32(x):
        return tf.cast(x, tf.float32)

    prefetchlen = int(1*1024*1024*1024 / (275 * 9 * 8)) # max 1 GB

    pd = pd.prefetch(prefetchlen)
    pd = pd.batch(prefetchlen//2) \
            .map(cast_to_float32,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch()
    pd = src.data.cache_dataset(pd, "pd", basepath=cache_path)
    pd = pd.prefetch(prefetchlen)

    ed = ed.prefetch(prefetchlen)
    ed = ed.batch(prefetchlen//2) \
            .map(cast_to_float32,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch()
    ed = src.data.cache_dataset(ed, "ed", basepath=cache_path)
    ed = ed.prefetch(prefetchlen)

    label = label.prefetch(prefetchlen)
    label = label.batch(prefetchlen//2) \
            .map(cast_to_float32,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch()
    label = src.data.cache_dataset(label, "label", basepath=cache_path)
    label = label.prefetch(prefetchlen)

    noise1 = src.data.random.uniform_dataset((100,))

    ds = tf.data.Dataset.zip((
        label,
        (pd, ed),
        (noise1,)
    ))
    ds : tf.data.Dataset = ds.shuffle(100000).batch(batchsize).prefetch(5)

    # create model
    import src.models.gan as gan

    gen = gan.BaseGenerator(depthlen, pd_maxdata, ed_maxdata)
    dis = gan.BaseDiscriminator(pd_maxdata, ed_maxdata)
    wd = gan.loss.WassersteinDistance(dis)
    gp = gan.loss.GradientPenalty(dis)

    gopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    dopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(logpath)

    # initialize and build once
    for label, real, noise in ds.take(1):
        out1 = gen([label, *noise,])
        out2 = dis([label, *real, *real,])
        out3 = wd([label, *real, *out1,])
        out4 = gp([label, *real, *out1,])

    # train function
    def train(dataset, gen, dis, wd, gp, gopt, dopt, epochs):
        dataset = dataset.repeat(epochs)
        for ii, (label, real, noise) in dataset.enumerate():
            if ii % 5 == 4:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(gen.trainable_weights)
                    fake = gen([label, *noise,])
                    distance = wd([label, *real, *fake,])
                    loss = distance

                tf.summary.scalar("Wasserstein Distance", distance, step=ii)

                grads = tape.gradient(loss, gen.trainable_weights)
                gopt.apply_gradients(zip(grads, gen.trainable_weights))

            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(dis.trainable_weights)
                    fake = gen([label, *noise,])
                    distance = wd([label, *real, *fake,])
                    penalty = gp([label, *real, *fake,])
                    loss = - distance + penalty

                tf.summary.scalar("Wasserstein Distance", distance, step=ii)
                tf.summary.scalar("Gradient Penalty", penalty, step=ii)
                tf.summary.scalar("Discriminator Loss", loss, step=ii)

                grads = tape.gradient(loss, dis.trainable_weights)
                dopt.apply_gradients(zip(grads, dis.trainable_weights))

    print("training ...")
    start = timeit.default_timer()
    with writer.as_default():
        try:
            train(ds, gen, dis, wd, gp, gopt, dopt, epochs)
        except KeyboardInterrupt:
            pass
        finally:
            writer.flush()
    end = timeit.default_timer()
    print("training time", end-start)

    # save model
    print("saving ...")
    for label, real, noise in ds.take(1):
        gen.predict([label, *noise,])
        dis.predict([label, *real, *real,])

    gen.save(os.path.join(savepath, "generator"))
    dis.save(os.path.join(savepath, "discriminator"))
    print("done ...")

    # 2. update "message"
    
    train_results = { "Done": f"Model successfully trained in {end-start} seconds" }
    message["training"].append(train_results)

    return message


# during development it might be practical 
# to check your code from CLI (command line interface)
def main():
    """
    Runs above-described methods from CLI
    (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta      
    elif args.method == 'predict':
        # [!] you may need to take special care in the case of args.files [!]
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'train':
        results = train(**vars(args))
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)

    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"deep_api.py method --help\" to get more info', 
                            dest='method')

    ## configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])                                      
    # normally there are no arguments to configure for get_metadata()

    ## configure arguments for predict()
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser]) 
    # one should convert get_predict_args() to add them in predict_parser
    # For example:
    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    ## configure arguments for train()
    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser]) 
    # one should convert get_train_args() to add them in train_parser
    # For example:
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    args = cmd_parser.parse_args()
    
    main()
