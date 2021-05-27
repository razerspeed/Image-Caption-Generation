import pandas as pd

import torchvision


import torch

from torch.autograd import Variable

pd.set_option('display.max_colwidth', None)






# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
import json

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

#loading model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()

resNet18Layer4 = resnet18._modules.get('layer4')


# creating hook
# https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

def get_vector(t_img):
    t_img = Variable(t_img)
    #     my_embedding = torch.zeros(1, 512, 7, 7)
    my_embedding = []

    def copy_data(model, model_input, model_output):
        my_embedding.append(model_output.data)

    h = resNet18Layer4.register_forward_hook(copy_data)
    resnet18(t_img)

    h.remove()
    return my_embedding





# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    # return 'Hello World'
    return jsonify({'status': 'Server 1 is UP ...'})


@app.route('/foo', methods=['POST'])
def foo():
    data = request.json
    image_data_torch = torch.tensor(data['image'])
    print(image_data_torch.shape)

    #get image embedding / features
    image_embedding = get_vector(image_data_torch)[0]
    to_return_image_embedding={'image_embedding': image_embedding.tolist()}

    # print(to_return_image_embedding.shape)

    return to_return_image_embedding


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
