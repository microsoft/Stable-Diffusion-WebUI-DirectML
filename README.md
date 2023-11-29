# DirectML Extension for Automatic1111's SD WebUI

This extension enables optimized execution of base Stable Diffusion models on Windows. Because DirectML runs across hardware, this means users can expect performance speed-ups on a broad range of accelerator hardware.

As a pre-requisite, the base models need to be optimized through [Olive](https://github.com/microsoft/Olive) and added to the WebUI's model inventory, as described in the Setup
section. The extension uses [ONNX Runtime](https://onnxruntime.ai/) and [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) to run inference against these models.

This preview extension offers DirectML support for compute-heavy uNet models in Stable Diffusion, similar to Automatic1111's sample TensorRT extension and NVIDIA's TensorRT extension. 

Stable Diffusion versions 1.5, 2.0 and 2.1 are supported.


## Getting Started

1. Follow instructions [here](https://github.com/microsoft/Olive/blob/main/examples/directml/stable_diffusion/README.md#setup) to setup Olive Stable Diffusion scripts to create optimized ONNX models.
2. Convert your SD model to ONNX, optimized by Olive, as described [here](https://github.com/microsoft/Olive/blob/main/examples/directml/stable_diffusion/README.md#conversion-to-onnx-and-latency-optimization). For the simplest case (the v1.5 safetensors downloaded during WebUI setup), the following command will suffice:
    ```
    python stable_diffusion.py --optimize
    ```
3. The optimized Unet model will be stored under `\models\optimized\[model_id]\unet` (for example `\models\optimized\runwayml\stable-diffusion-v1-5\unet`).Copy this over, renaming to match the filename of the base SD WebUI model, to the WebUI's `models\Unet-dml` folder.
4. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
5. Back in the main UI, select the DML Unet model from the sd_unet dropdown menu at the top of the page, and get going.
<!-- end numbered list -->
</ol>

## Notes

While the models have been pre-optimized by Olive, DirectML can further fine-tune them for performance on session activation. Such optimizations, however,     assume fixed dimensions: batch size/image resolution for all jobs in the session.A new session would be initiated if these assumptions no longer hold true. This incurs a start overhead that will lower performance for the first job with the new dimensions. Fixed dimension optimizations are enabled by default, but can be disabled in Settings → DirectML → Enable DML Unet Static Dimensions. For Hires Fix, disabling this is recommended.

Image dimensions need to be specified as multiples of 64.

Stable Diffusion XL (and by extension, Refiner) is not supported at this time, nor are LoRA/ControlNet.

For non-CUDA compatible GPU, launch the Automatic1111 WebUI by updating the webui-user.bat as follows
```
set COMMANDLINE_ARGS=--lowvram --precision full --no-half --skip-torch-cuda-test
```
Once started, the extension will automatically execute the uNet path via DirectML on the available GPU.
 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
