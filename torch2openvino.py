import openvino as ov
import torch
import os
from model import enhance_color

def load_pretrain_network(ckpt, device):
    net = enhance_color().to(device)
    net.load_state_dict(torch.load(os.path.join(ckpt), map_location=torch.device(device))['state_dict'])
    return net


ckpt = 'weights/enhance_color-llie-ResCBAM_g.pkl'
MODEL_NAME='OV_enhance_color-llie-ResCBAM_g-256'

network = load_pretrain_network(ckpt, 'cpu')
network(torch.randn(8, 3, 256, 256))
network.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(network, dummy_input, MODEL_NAME+'.onnx' )
ov_model = ov.convert_model(MODEL_NAME+'.onnx')

ov.save_model(ov_model, MODEL_NAME+'.xml', compress_to_fp16=False )

print(ov_model)