import os
import torch
from core.model import create_model
from .storage import load_model, checkpoint_dir


def shape_inference(onnx_file, inputs):
    import onnx_tool
    from onnx_tool import create_ndarray_f32
    m = onnx_tool.Model(onnx_file)
    input_shapes = {}
    for i in inputs:
        input_shapes[i[0]] = create_ndarray_f32(i[1])
    m.graph.shape_infer(input_shapes)
    m.graph.profile()
    m.graph.print_node_map()
    m.save_model(onnx_file)


def export_onnx(cfg, epoch, inputs, output_names, opset=11):
    ckp_dir = checkpoint_dir(cfg)
    model_cfg = cfg.model
    ckp_file = os.path.join(ckp_dir, 'epoch_{:05}.pth'.format(epoch))
    model = create_model(model_cfg)
    load_model(ckp_file, model)
    model.eval()
    if model.onnx_forward is not None:
        model.forward = model.onnx_forward
    input_tensors = []
    input_names = []
    for i in inputs:
        input_names.append(i[0])
        input_tensors.append(torch.rand((i[1])))
    out_file = os.path.join(ckp_dir, "epoch_{:05}.onnx".format(epoch))
    torch.onnx.export(model,
        tuple(input_tensors),
        out_file,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names = input_names,
        output_names = output_names)
    shape_inference(out_file, inputs)
