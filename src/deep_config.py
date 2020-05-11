# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')
REPORTS_DIR = os.path.join(IN_OUT_BASE_DIR, 'reports')

BASE_DATA_DIR = os.path.join(BASE_DIR, 'data')
BASE_MODELS_DIR = os.path.join(BASE_DIR, 'models')
BASE_REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

if IN_OUT_BASE_DIR != BASE_DIR:
    # create symlinks
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(DATA_DIR)
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(MODELS_DIR)
    if not os.path.isdir(REPORTS_DIR):
        raise FileNotFoundError(REPORTS_DIR)

    if os.path.exists(BASE_DATA_DIR):
        os.unlink(BASE_DATA_DIR)
    if os.path.exists(BASE_MODELS_DIR):
        os.unlink(BASE_MODELS_DIR)
    if os.path.exists(BASE_REPORTS_DIR):
        os.unlink(BASE_REPORTS_DIR)

    os.symlink(DATA_DIR, BASE_DATA_DIR)
    os.symlink(MODELS_DIR, BASE_MODELS_DIR)
    os.symlink(REPORTS_DIR, BASE_REPORTS_DIR)

    print(f"symlinks to {IN_OUT_BASE_DIR} created")

# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select a file for the prediction"
    )

    # to be able to provide an URL for prediction
    urls = fields.Url(
        required=False,
        missing=None,
        description="Provide an URL of the data for the prediction"
    )
    
    # an input parameter for prediction
    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for the prediction"
    )

# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for training"
    )

    dataset = fields.Str(
            required=False,
            missing="run01",
            description="processed dataset which should be used for training"
    )

    save_name = fields.Str(
            required=False,
            missing="gan",
            description="save name which should be used in predict"
    )

    cache_path = fields.Str(
            required=False,
            missing="/home/tmp/koepke/cache",
            description="cache basepath for dataset"
    )

    epochs = fields.Integer(
        required=False,
        missing=1,
        description="number of epochs for training"
    )

    batchsize = fields.Integer(
        required=False,
        missing=1024,
        description="number of epochs for training"
    )

