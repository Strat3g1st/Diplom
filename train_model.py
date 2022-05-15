import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
import matplotlib.pyplot as plt
from decoder_32 import CNNNet


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


def train(model, optimizer, loss_fn, train_data_loader, epochs=20):
    total_steps = len(train_data_loader)
    for epoch in range(epochs):
        for i, (images, masks) in enumerate(train_data_loader, 1):
            images = images.type(torch.FloatTensor)
            images = images.reshape(images.shape[0], images.shape[3], images.shape[1], images.shape[2])
            masks = masks.type(torch.FloatTensor)
            # masks = masks.reshape(masks.shape[0], masks.shape[3], masks.shape[1], masks.shape[2])
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = loss_fn(torch.max(outputs, 1)[0].float(), masks)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # if (i) % 100 == 0:
            print("Epoch [" + str(epoch + 1) + "/" + str(epochs) + "], Step ["
                  + str(i) + "/" + str(total_steps) + "], Loss: " + str(loss.item()))
            losses.append(loss.item())


def sort_special(method, lst, len):
    lst_method = []
    while method < len:
        lst_method.append(lst[method])
        method += 3
    return lst_method


torch.cuda.device('cuda')
torch.cuda.empty_cache()
model = CNNNet()
# model.load_state_dict(torch.load("model_unet_256_x3"))
# model.eval()
model.cuda()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

learning_rate = 1e-4
batch_size = 50
epochs = 50

N = 72000
path_to_images = "C:\\Users\\Admin\\Documents\\dataset_object\\images\\"
path_to_masks = "C:\\Users\\Admin\\Documents\\dataset_object\\targets\\"
inputs = os.listdir(path_to_images)
targets = os.listdir(path_to_masks)
inputs = sorted(inputs, key=lambda x: int(os.path.splitext(x)[0]))
targets = sorted(targets, key=lambda x: int(os.path.splitext(x)[0]))
inputs = inputs[:int(N/2)]
targets = targets[:int(N/2)]
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
training_dataset = SegmentationDataSet(inputs=inputs_HII, targets=targets_HII, transform=None)
training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("start learning")
train(model, optimizer, loss_fn, training_dataloader, epochs=epochs)

torch.save(model.state_dict(), "model_decoder_32_x5_HII")

plt.plot(range(0, len(losses)), losses, label='losses value')
plt.legend()
plt.show()
