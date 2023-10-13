import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim 
import evolve
import pdb
import os
import datetime

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    steps = len(train_loader)     
    for step,(data, target) in enumerate(train_loader, start=0):
        # step -> batchIndex    last_step = len(loader) / batchsize 
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        running_loss += loss.item() / steps
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}, loss of this epoch: {:.6f}'.format(
        epoch, running_loss))
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nAccuracy: {}/{} ({:.6f})\n'.format(
        correct, len(test_loader.dataset),
        1. * correct / len(test_loader.dataset)))
    fitness = 1. * correct / len(test_loader.dataset)
    return fitness

def trainGroup(group):
    generation = []
    for index, model in enumerate(group, start = 0):
        print("This is {} individual in this generation".format(index+1))
        batch_size = model.gene['batch_size']
        lr = model.gene['lr']
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=False,
                transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                batch_size=batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr)
        for epoch in range(1, 21):
            train(model, device, train_loader, optimizer, epoch)
            model.gene['fitness'] = test(model, device, test_loader)
        generation.append(model.gene)
    return generation

def main():
    # args:
    num_generations = 10
    num_individuals = 20
    survive_percent = 0.25
    mutate_prob = 0.2

    generation = evolve.generateGeneration(num_individuals) # a generation genes
    os.remove('record.txt')
    file = open('record.txt', 'a')
    file.write("Start time is: " + str(datetime.datetime.now()))
    file.write("\nThe initial generation's genes are :\n")
    for gene in generation:
        file.write(str(gene) + '\n')
    file.close()
    for gens in range(1, num_generations+1):
        print("This is {} generation of the evolution".format(gens))
        group = evolve.createGroup(generation) # actual individuals
        generation = trainGroup(group) # update fitness for all individual
        generation = evolve.evolve(generation, survive_percent, mutate_prob, False if gens != num_generations else True)
        file = open('record.txt', 'a')
        for ID, gene in enumerate(generation, start = 1):
            file.write("generation:{}, individual:{}, fitness:{}\n".format(gens, ID, gene['fitness']))
        file.write("The best individual is:\n" + str(generation[0]))
        file.write("\nEnd time is: " + str(datetime.datetime.now()))
        file.close()
if __name__ == '__main__':
    main()

