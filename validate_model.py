import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
import matplotlib.pyplot as plt
from decoder_256 import CNNNet


class SegmentationDataSet(Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        # y = np.expand_dims(y[:, :, 0], axis=2)
        y = y[:, :, 0]
        y[y == 255] = 1

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x, y


losses = []
good = []


def validate(model, loss_fn, train_data_loader, min_fn, epochs=1):
    total_steps = len(train_data_loader)
    for epoch in range(epochs):
        for i, (images, masks) in enumerate(train_data_loader, 1):
            images = images.type(torch.FloatTensor)
            images = images.reshape(images.shape[0], images.shape[3], images.shape[1], images.shape[2])
            masks = masks.type(torch.FloatTensor)
            images = images.cuda()
            masks = masks.cuda()

            outputs = model.forward(images)
            loss = loss_fn(torch.max(outputs, 1)[0].float(), masks)

            # if (i) % 100 == 0:
            print("Epoch [" + str(epoch + 1) + "/" + str(epochs) + "], Step ["
                  + str(i) + "/" + str(total_steps) + "], Loss: " + str(loss.item()))
            losses.append(loss.item())
            if loss.item() < min_fn:
                good.append(loss.item())


def sort_special(method, lst, len):
    lst_method = []
    while method < len:
        lst_method.append(lst[method])
        method += 3
    return lst_method


torch.cuda.device('cuda')
torch.cuda.empty_cache()
model = CNNNet()
model.load_state_dict(torch.load("model_decoder_256_x5"))
model.eval()
model.cuda()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

learning_rate = 1e-4
batch_size = 1
epochs = 1
min_fn = 0.1

N = 72000
path_to_images = "C:\\Users\\Admin\\Documents\\dataset_object\\images\\"
path_to_masks = "C:\\Users\\Admin\\Documents\\dataset_object\\targets\\"
inputs = os.listdir(path_to_images)
targets = os.listdir(path_to_masks)
inputs = sorted(inputs, key=lambda x: int(os.path.splitext(x)[0]))
targets = sorted(targets, key=lambda x: int(os.path.splitext(x)[0]))
inputs = inputs[int(N/2):]
targets = targets[int(N/2):]
inputs = [path_to_images + file for file in inputs]
targets = [path_to_masks + file for file in targets]
CRA = 0
DFN = 1
HII = 2
inputs_CRA = sort_special(CRA, inputs, int(N/2))
targets_CRA = sort_special(CRA, targets, int(N/2))
inputs_DFN = sort_special(DFN, inputs, int(N/2))
targets_DFN = sort_special(DFN, targets, int(N/2))
inputs_HII = sort_special(HII, inputs, int(N/2))
targets_HII = sort_special(HII, targets, int(N/2))
training_dataset = SegmentationDataSet(inputs=inputs, targets=targets, transform=None)
training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.BCEWithLogitsLoss().cuda()

print("start validating")
validate(model, loss_fn, training_dataloader, min_fn, epochs=epochs)

# plt.plot(range(0, len(losses)), losses, label='losses value')
# plt.legend()
# plt.show()
print(len(good)/len(inputs))

# 0.29323529411764704 [1/10, 100ep]
# 0.31436274509803924 [1/10, 30ep]
# 0.2813480392156863 [1/10, 40ep]
# 0.41422619047619047 [2/10, 100ep]
# 0.520719696969697 [3/10, 50ep]
# ------------------------------------
# UNet
# 0.29441666666666666 [1/10 datatrain, 100ep]
# 0.4196666666666667 [2/10 datatrain, 100ep]
# 0.52075 [3/10 datatrain, 100ep]
# 0.5915 [4/10 datatrain, 100ep]
# 0.6274166666666666 [5/10 datatrain, 100ep]
# ------------------------------------------
# decoder256: 0.6116666666666667 [5/10 datatrain, 50ep]
# decoder128:
