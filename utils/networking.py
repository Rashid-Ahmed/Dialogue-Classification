import socket
import sys
import torch
import copy
import numpy as np
from utils.inference_utils import get_predictions_JSON

def start_tcp_server(IP, PORT, MODEL_TYPES, MODELS, TOKENIZER, device):
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (IP, PORT)
        sock.bind(server_address)
        
        sock.listen(1)
        connection, _ = sock.accept()

        while True:
            sentence = connection.recv(64000).decode('utf-8')
            if not sentence or len(sentence) <= 2 or sentence == "connection closed":
                break
            
            json_predictions = get_predictions_JSON(MODEL_TYPES, MODELS, TOKENIZER, sentence, device)
            
            connection.sendall(json_predictions.encode())   
            
                
        connection.close()