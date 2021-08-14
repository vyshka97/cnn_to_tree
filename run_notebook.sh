#!/usr/bin/env bash

NTB_DIR=$1

jupyter notebook --allow-root --notebook-dir=${NTB_DIR} --ip=0.0.0.0
