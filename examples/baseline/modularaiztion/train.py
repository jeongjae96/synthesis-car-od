import torch

import numpy as np
from tqdm import tqdm

from config import CFG

def validation(
    model,
    val_loader,
    device
):
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(iter(val_loader)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output in outputs:
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
    

def train(
    model, 
    train_loader,
    val_loader,
    device
):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = 9999999
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for images, targets in tqdm(iter(train_loader)):
            images = [img.to(device) for img in images]
            targets = [{k : v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())

        if scheduler is not None:
            scheduler.step()
        
        tr_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')
        
        if best_loss > tr_loss:
            best_loss = tr_loss
            best_model = model
    
    return best_model