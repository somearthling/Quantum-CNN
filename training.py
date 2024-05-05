'''
Contains functions to generate the quantum circuit using architecture.py, evaluate the gradient and cost function,
and train the model.
'''
from functools import partial
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.encoders import encode_base64
import json

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

import architecture as arch

def circuit(image, wires, params, conv_params, pool_params, dense_params):
    '''
    Generates the quantum circuit for the given image and params.
    '''

    # Encode image
    qml.AmplitudeEmbedding(features=image, wires=wires, normalize=True)

    arch.QCNN(params, wires=wires, conv_params=conv_params, pool_params=pool_params, dense_params=dense_params)

    return qml.probs(wires=arch.QCNN.out_wires(wires, conv_params, dense_params))

@partial(jax.jit, static_argnums=(0, 1, 5, 6, 7))
def gradient_and_cost(map_qnode, map_jacobian, images, labels, params, conv_params, pool_params, dense_params):
    '''
    Evaluates the gradient and cost function for the current circuit parameters given a batch of images and labels. 
    Uses vectorization over the batch.
    The cost function is cross-entropy loss, that is given by -log(correct_class_probability).
    The gradient of the cost function is given by -1/prob(correct_class) * dprob(correct_class)/dparams.
    '''
    wires = range((np.ceil(np.log2(len(images[0])))).astype(int)) # number of input qubits
    probs = map_qnode(images, wires, params, conv_params, pool_params, dense_params)
    jacobians = map_jacobian(images, wires, params, conv_params, pool_params, dense_params)

    # get average gradients and cost over the batch from probs and jacobians
    correct_probs = jnp.take_along_axis(probs, labels[:, None], axis=1).squeeze()
    correct_jacobians = jnp.take_along_axis(jacobians, labels[:, None, None], axis=1).squeeze()

    epsilon = 1e-10  # small constant to prevent division by zero
    gradients = -jnp.mean(correct_jacobians / (correct_probs[:, None] + epsilon), axis=0)
    cost = -jnp.log(correct_probs).mean()

    return gradients, cost


def send_email(subject, message, filename=None):
    '''
    Sends an email with the given subject and message.
    '''

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    from_email = config["email"]
    to_email = config["to"]
    password = config["password"]

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    msg.attach(MIMEText(message, 'plain'))

    if filename:
        attachment = open(filename, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {filename}')
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

def handle_exception(exc_type, exc_value, exc_traceback):
    '''
    Handles exceptions by sending an email with the exception message.
    '''
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    send_email(
        subject="An error occurred",
        message=str(exc_value),
    )
