def get_weights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from decoder_32 import CNNNet
model = CNNNet()
print(get_weights(model))

from decoder_64 import CNNNet
model = CNNNet()
print(get_weights(model))

from decoder_128 import CNNNet
model = CNNNet()
print(get_weights(model))

from decoder_256 import CNNNet
model = CNNNet()
print(get_weights(model))

from UNet256 import CNNNet
model = CNNNet()
print(get_weights(model))

from UNet256_dropout import CNNNet
model = CNNNet()
print(get_weights(model))
