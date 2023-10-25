import torch
import torch.nn
from torch import optim
from seq_dataset import WaterDataSet
from torch.utils.data import DataLoader
from model import Transformer
from tqdm import tqdm
import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = WaterDataSet(root_dir=cfg.train_file_path,
                       source_len=cfg.input_source_seq_len,
                       target_len=cfg.input_target_seq_len,
                       output_len=cfg.output_seq_len,
                       step_ahead=cfg.number_step_ahead,
                       data_col=cfg.input_col,
                       target_col=cfg.target_col)

data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

transformer = Transformer(
    input_feature=cfg.input_feature,
    output_feature=cfg.output_feature,
    embed_dim=cfg.embed_dim,
    layer_num=cfg.layer_num,
    expansion_dim=cfg.expansion_dim,
    head_num=cfg.head_num,
    device=cfg.device,
    input_seq_len=cfg.input_seq_len,
    output_seq_len=cfg.output_seq_len).to(cfg.device)

optimizer = optim.Adam(transformer.parameters(), lr=cfg.learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

criterion = torch.nn.MSELoss()


def run_train():
    step = 0

    for epoch in range(cfg.epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

        transformer.train()
        losses = []

        for batch_idx, (source_, target, label) in loop:
            # Get input and targets and get to cuda
            source = source_.to(device)
            target = target.to(device)
            label = label.to(device)
            # print(in_put.shape, target.shape)

            # Forward prop
            output = transformer(source, target)

            optimizer.zero_grad()

            loss = criterion(output[:, :, -1], label[:, :, -1])
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            step += 1
            print(loss.item())
            loop.set_description(f'epoch[{epoch}/{cfg.epochs}]')
            loop.set_postfix(loss=loss.item())

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)


if __name__ == "__main__":
    run_train()
