import os
import onnx
from onnx2pytorch import ConvertModel
from main import train

train()
for idx, opponent in enumerate(os.listdir("./checkpoints/curriculum-actors")):
    if idx < 7:
        continue
    print(opponent)
    checkpoint_path = "checkpoints/stage_2.pt"
    net = ConvertModel(onnx.load(os.path.join("checkpoints/curriculum-actors", opponent)), experimental=True)
    if idx > 2:
        checkpoint_path = "curriculum.pt"
    train(opponent=net, checkpoint_path=checkpoint_path)
    # train()
