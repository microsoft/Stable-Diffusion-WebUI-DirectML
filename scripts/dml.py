# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import numpy as np
import copy
import gradio as gr

import onnxruntime as ort
import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal


# Structure of this extension adapted from Automatic's reference Unet extension for TensorRT:
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt, and by extension:
# https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT

class DMLUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[DML] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        if shared.sd_model.is_sdxl:
            raise ValueError(
                    "SD XL models are not supported with this version of the DirectML extension."
                )
        return DMLUnet(self.filename)


ort_to_torch = {
    "tensor(float32)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(int32)": torch.int32,
}

input_to_ort_names_map = {
    "x": "sample",
    "timesteps": "timestep",
    "context":"encoder_hidden_states"
}




class DMLUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.session = None
        self.sess_options = None
        self.output_name = None
        self.buffers = None
        self.dimensions = {}
        self.static_dimension_dims = True

    def dimensionschanged(self,feed_dict):
        if ((self.dimensions["width"] != feed_dict['x'].shape[-1]) or
        (self.dimensions["height"] != feed_dict['x'].shape[-2]) or
        (self.dimensions["batch_size"] != feed_dict['x'].shape[-4])):
            gr.Info("Dimensions changed, restarting DML inference session, first job may be slow. \
                    Select dynamic dimensions from Settings if you expect to change image size/batch size frequently")
            return True

        return False

    def infer(self, feed_dict):
        new_profile = shared.opts.data.get("directml_static_dims", True)
        if (self.session is None or
            new_profile!= self.static_dimension_dims or
            (self.static_dimension_dims and self.dimensionschanged(feed_dict))):

            self.dimensions["width"] = feed_dict['x'].shape[-1]
            self.dimensions["height"] = feed_dict['x'].shape[-2]
            self.dimensions["batch_size"] = feed_dict['x'].shape[-4]
            self.dimensions["channel_size"] = feed_dict['x'].shape[-3]

            self.initsession()

        self.static_dimension_dims = new_profile
        self.buffers = {}

        for name, tensor in feed_dict.items():
            inputname = input_to_ort_names_map[name]
            self.buffers[inputname] = copy.deepcopy(tensor.cpu().numpy().astype(np.float16))


        self.output_name = self.session.run(None, self.buffers)

    def forward(self, x, timesteps, context, *args, **kwargs):
        feed_dict = {
            "x": x,
            "timesteps": timesteps,
            "context": context,
        }

        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError(
                    "Input shape must be divisible by 64 in both dimensions."
                )

        self.infer(feed_dict)

        return torch.from_numpy(self.output_name[0]).to(dtype=x.dtype, device = devices.device)


    def activate(self):
        print("DML activation delayed to inference time")

    def initsession(self):
        self.sess_options = ort.SessionOptions()
        ort.set_default_logger_severity(3)
        self.sess_options.enable_mem_pattern = False
        providers = [
            (
                "DmlExecutionProvider"
            )
        ]
        #sess_options.enable_profiling = True

        if shared.opts.data.get("directml_static_dims", False):
            height = self.dimensions["height"]
            width = self.dimensions["width"]
            batch_size = self.dimensions["batch_size"]
            channels_size = self.dimensions["channel_size"]
            self.sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size)
            self.sess_options.add_free_dimension_override_by_name("unet_sample_channels", channels_size)
            self.sess_options.add_free_dimension_override_by_name("unet_sample_height", height)
            self.sess_options.add_free_dimension_override_by_name("unet_sample_width", width)
            self.sess_options.add_free_dimension_override_by_name("unet_time_batch", batch_size)
            self.sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size)
            self.sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

        print("activating DML Unet")
        self.session = ort.InferenceSession(self.filename, providers=providers, sess_options=self.sess_options)

    def deactivate(self):
        devices.torch_gc()


def list_unets(unet_list):
    dml_dir = os.path.join(paths_internal.models_path, 'Unet-dml')
    candidates = list(shared.walk_files(dml_dir, allowed_extensions=[".onnx"]))
    for filename in sorted(candidates, key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]

        opt = DMLUnetOption(filename, name)
        unet_list.append(opt)

def on_ui_settings():
        section = ('directml', "DirectML")
        shared.opts.add_option("directml_static_dims", shared.OptionInfo(
        True, "Enable DML Unet Static Dimensions", gr.Checkbox, {"interactive": True}, section=section))

def on_ui_tabs():
    with gr.Blocks() as dml_interface:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with open(
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "help.md"),
                    "r",
                    encoding='utf-8',
                ) as f:
                    gr.Markdown(elem_id="dml_info", value=f.read())

    return [(dml_interface, "DirectML", "directml")]

script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)


