from tqdm import tqdm
import torch
from torch import optim
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    now_batch = 0
    
    total_batch = len(train_loader.dataset)
    for b in train_loader:
        # 帮助了解进展（毕竟batch很大）
        now_batch += 1
        if now_batch%1000 ==0:
            print('batch now:{},total batch:{}'.format(now_batch,total_batch))
        
        ids1, m1, ids2, m2, y = b
        loss = model.get_loss(ids1.to(model.device), ids2.to(model.device), m1.to(model.device), m2.to(model.device),
                              y.to(model.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
        
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss.item()


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for b in data_loader:
            ids1, m1, ids2, m2, y = b
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.argmax(
                model(ids1.to(model.device), ids2.to(model.device), m1.to(model.device), m2.to(model.device)),
                dim=1).cpu().tolist())
            loss = model.get_loss(ids1.to(model.device), ids2.to(model.device), m1.to(model.device),
                                  m2.to(model.device), y.to(model.device))
            total_loss += loss
        avg_loss = total_loss / len(data_loader.dataset)
    print(classification_report(y_true, y_pred))
    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    test_loss = eval_loss(model, test_loader)
    test_losses.append(test_loss)  # loss at init
    for epoch in tqdm(range(epochs), desc='Epoch', leave=False):
        model.train()
        epoch_train_losses = train(model, train_loader, optimizer)
        train_losses.append(epoch_train_losses)
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)

    plot_train_curve(train_losses, test_losses)
    return train_losses, test_losses


def plot_train_curve(train_losses, test_losses):
    # 生成横轴数据（epochs）
    train_epochs = torch.range(1, len(train_losses))
    test_epochs = torch.range(0, len(test_losses) - 1)
    # 绘制训练集损失折线
    plt.plot(train_epochs, train_losses, color='blue', label='Train Loss')

    # 绘制测试集损失折线
    plt.plot(test_epochs, test_losses, color='red', label='Test Loss')

    # 设置图表标题和轴标签
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 添加图例
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Roberta_training_curve_{timestamp}.png'

    # 先save在show才能正确保存
    plt.savefig(filename)
    # 显示图表
    plt.show()

