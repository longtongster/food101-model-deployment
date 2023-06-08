from model import create_effnetb2_model
from functools import partial
from timeit import default_timer
import PIL
import torch
import gradio as gr


def predict(img, model, transform, class_names, device):
    start_time = default_timer()
    # send the model to device
    model.to(device)
        
    # Transform the image, create a batch dimension and send to device
    img = transform(img).unsqueeze(dim=0).to(device)
    
    # in inference model we set the model to eval
    model.eval()
    
    # set the model to inference model as well
    with torch.inference_mode():
        # Forward pass of the model
        logits = model(img)
        probs = logits.softmax(dim=1).squeeze(0)

        class_and_probs = {class_names[i]:p.item() for i,p in enumerate(probs)}
        #print(class_and_probs)
    
    pred_time = default_timer() - start_time
    
    return class_and_probs, pred_time


if __name__ == "__main__":
    # get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # read the class names
    with open("./class_names.txt") as f:
        class_names = list()
        for class_name in f.readlines():
            class_names.append(class_name)

    num_classes = len(class_names)
    
    # create an effnetb2 model and transform
    model, transform = create_effnetb2_model(num_classes, feature_extracting=True)
    
    # load trained weights from the artifact
    model_artifact = torch.load("./effnetb2_fine-tuning_adam_lr_0.0001_40.0%_data_10_epochs.pth", map_location=torch.device(device))
    trained_model_weights = model_artifact['model_state_dict']
    model.load_state_dict(trained_model_weights)
    
    predict_ = partial(predict, model=model, transform=transform, class_names=class_names, device=device)
        

    title = "Foodvision Mini"
    description = "EfficientNetB feature extractor to classify images as pizza, steak or sushi"
    article = "Created at 09 PyTorch model deployment"

    # Create the Gradio 
    demo = gr.Interface(fn=predict_, 
                 inputs=gr.Image(type="pil"),
                 outputs = [gr.Label(num_top_classes=5, label="Prediction"),
                            gr.Number(label="Prediction time (s)")],
                 #examples=example_list,
                 title=title,
                 description=description, 
                 article=article)

    demo.launch(debug=False, # print errors locally
                share=False, # generate a publically sharable URL
                #server_name="0.0.0.0",
                server_port=8080# set the port you want gradio to run on
               )
