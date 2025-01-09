from utils.imports import *
from datasets.CustomDataset import TyphoonDataset
from modules.network import get_loss_function, get_scheduler
from modules.components.get_model import get_model
import matplotlib.pyplot as plt
import torchvision.utils

def train(config):
    random_seed(config['seed'])

    if not torch.cuda.is_available():  # CUDA가 사용 가능하지 않으면
        raise RuntimeError("CUDA가 사용 가능하지 않습니다.") 
    print("CUDA가 사용 가능합니다.")
    device = 'cuda'
    root_directory = config['dataroot']
    csv_train_path = root_directory + "/data_before_2024.csv"
    csv_test_path = root_directory + "/data2024.csv"

    train_dataset = TyphoonDataset(csv_file=csv_train_path, root_dir=root_directory, config=config, visualize_aug= False)
    val_dataset = TyphoonDataset(csv_file=csv_test_path, root_dir=root_directory, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False)  
    
    criterion = get_loss_function(config=config)
    
    model = get_model(config)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    
    # 마지막 분류기 층 수정
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])  # 바꿔도 되고 
    scheduler = get_scheduler(config=config, optimizer=optimizer, loader_length=len(train_loader))
    start_epoch = 0

    if config['continue'] == True:
        checkpoint = torch.load(f"checkpoints/{config['name']}.pth")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    model.to(device)

    ###train
    best_model = None
    summary_iter = 0

    best_val_loss = 1000000.0
    best_acc = 0
    best_epoch = 0
    
    # 손실값 기록용 리스트 추가
    train_losses = []
    val_losses = []
    
    accumulation_steps = 8  # 4 * 8 = 32의 효과
    
    for epoch in range(start_epoch, config["epoch"]):
        print(f"Epoch {epoch}/{config['epoch']}")
        model.train()
        train_loss = 0.0
        total_samples = 0
        idx = 0

        pbar = tqdm(iter(train_loader))
        for batch in pbar:
            # 첫 번째 배치에서만 샘플 이미지 저장
            if idx == 0:
                # 8개의 샘플 이미지 선택
                sample_images = batch['IMAGE'][:8]
                sample_labels = batch['LABEL'][:8]
                
                # 이미지 그리드 생성 및 저장
                grid = torchvision.utils.make_grid(sample_images, nrow=4, normalize=True)
                plt.figure(figsize=(15, 8))
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.title(f'Training Samples (Epoch {epoch})\nLabels: {sample_labels.cpu().numpy()}')
                plt.axis('off')
                plt.savefig(f"checkpoints/{config['name']}_samples_epoch{epoch}.png")
                plt.close()
            
            image, label = batch['IMAGE'].to(device), batch['LABEL'].to(device)  # 'IMAGE' 및 'LABEL' 키 접근
            output = model(image)
            loss = criterion(output, label)  
            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            train_loss += loss.item() * accumulation_steps  # Rescale loss for logging
            total_samples += label.size(0)

            # Gradient Accumulation
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            idx += 1
            summary_iter += 1
            current_lr = optimizer.param_groups[0]["lr"]  # 현재 학습률 가져오기
            pbar.set_postfix_str(
                f"Ls: {train_loss / (idx + 1):.4f}, "  # 배치 수로 나누기
                f"LR: {current_lr:.6f}"
            )

        # Handle any remaining gradients at the end of epoch
        if idx % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Add average training loss print
        avg_train_loss = train_loss / (idx + 1)
        print(f"Training Loss: {avg_train_loss:.4f}")

        _val_loss = validate(config, device, criterion, val_loader, model)
        
        # 손실값 기록
        train_losses.append(avg_train_loss)
        val_losses.append(_val_loss)
        
        if _val_loss < best_val_loss:
            best_val_loss = _val_loss
            best_model = deepcopy(model.state_dict())
            best_epoch = epoch
            

        
    model.load_state_dict(best_model)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": best_epoch + 1,
    }, f"checkpoints/{config['name']}.pth")
    
    # 학습 완료 후 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # 그래프 저장
    plt.savefig(f"checkpoints/{config['name']}_loss_curve.png")
    plt.close()

    return model, best_epoch, best_acc


def validate(config, device, criterion, val_loader, model):
    model.eval()
    val_loss = 0.0
    # Tracking group-wise losses
    group_losses = {}
    group_counts = {}
    idx = 0
    pbar = tqdm(iter(val_loader))
    
    with torch.no_grad():
        for batch in pbar:
            image, label = batch['IMAGE'].to(device), batch['LABEL'].to(device)
            output = model(image)
            loss = criterion(output, label)

            val_loss += loss.item()
            # Group-based accumulation
            per_sample_loss = loss.item() / label.size(0)
            for i in range(label.size(0)):
                grade_val = int(batch['GRADE'][i].item())
                if grade_val not in group_losses:
                    group_losses[grade_val] = 0.0
                    group_counts[grade_val] = 0
                group_losses[grade_val] += per_sample_loss
                group_counts[grade_val] += 1
            idx += 1

    # Print overall validation loss
    print(f"Validation Loss: {val_loss / (idx + 1):.4f}")
    # Print group-based losses
    for g in sorted(group_losses.keys()):
        print(f"Grade {g} Loss: {group_losses[g]/group_counts[g]:.4f}")
    return val_loss / (idx + 1)



if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configs', 'STAGE1.yaml')

    with open(config_path, 'r') as file:  
        configs_stage1 = yaml.safe_load(file)

    config = configs_stage1
    model, best_val_loss, best_acc = train(config)
    print(f"Best Validation Loss: {best_val_loss}, Best Accuracy: {best_acc}")
