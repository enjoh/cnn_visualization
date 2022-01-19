import torch
import torchvision.models as models
from DeconvNet import DeconvNet
import sys
from utils import get_image, visualize_results, get_label_dict

if __name__ == '__main__':
    model = models.vgg19(pretrained=True)
    model.eval()

    deconv_model = DeconvNet(from_conv=model)
    deconv_model.eval()

    img_in = get_image("cat.jpeg")
    out = model(img_in)

    pred = torch.argmax(out).item()
    idx2label = get_label_dict()
    print(f"Predicted label: {idx2label[pred]} ({out.detach().numpy().max()})")
    # print("ConvNet=", model.features)
    # print("DeconvNet=", deconv_model.deconv_layers)

    layer_idx = int(input("Enter layer to visualize [" + ", ".join([str(i) for i in deconv_model.conv_indices]) + "]: "))
    print(f"Entered {layer_idx}")

    if layer_idx not in deconv_model.conv_indices:
        print(f"Wrong layer index, re-run and try again!")
        sys.exit(0)

    out_deconv = deconv_model(out, layer_idx)
    visualize_results(img_in, out_deconv, layer_idx)

