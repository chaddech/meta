# meta

Runs on `wideresnet.py` from https://github.com/xternalz/WideResNet-pytorch

### How to run
`python get_intermediate_wideresnet.py` to get intermediate layer data

`python consolidate_cifar_inter_outputs.py` in order to consolidate data

`python base_classify_multi_pre_bn_last_conv_pen_conv_output_balanced.py` to run classifier