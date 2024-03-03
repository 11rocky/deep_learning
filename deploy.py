import argparse
from core.api.export import export_onnx
from core.api.utils import init_cfg


def main():
    opt = argparse.ArgumentParser("main")
    opt.add_argument("--config", type=str, required=True, help="config file")
    opt.add_argument("--epoch", type=int, required=True, help="epoch to export")
    opt.add_argument("--inputs", type=str, nargs="+", required=True,
                     help="input name and shape, eg. image:1,3,224,224 mask:1,1,224,224")
    opt.add_argument("--outputs", type=str, nargs="*", help="output names eg. output1 output2")
    args = opt.parse_args()
    cfg = init_cfg(args.config)
    inputs = []
    for i in args.inputs:
        tmp = i.split(":")
        name = tmp[0]
        shape = [int(x) for x in tmp[1].split(",")]
        inputs.append((name, tuple(shape)))
    outputs = ["output"] if args.outputs is None else args.outputs
    export_onnx(cfg, args.epoch, inputs, outputs)


if __name__ == '__main__':
    main()
