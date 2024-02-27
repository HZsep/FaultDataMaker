import torch
from network.generator import Generator
from network.discriminator import Discriminator
from utils.utils import getConfig, json_loader
from dataset import get_loader
import random
import numpy as np
from loss import Dis_loss, Gen_loss
import os
from utils.data_utils import inverse


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        configs = getConfig()
        train_configs, num_classes, backbone = configs['train'], int(configs['num_classes']), configs['backbone']
        gen_optim, dis_optim, feature_nums = configs['gen_optim'], configs['dis_optim'], configs['feature_nums']
        gen_lr, gen_wd = gen_optim['lr'], gen_optim['weight_decay']
        dis_lr, dis_wd = dis_optim['lr'], dis_optim['weight_decay']
        self.lambda_gp = float(configs['WGAN-GP']['lambda_gp'])
        dis_betas, gen_betas = configs['dis_optim']['betas'], configs['gen_optim']['betas']

        load_if = int(input("If load? 0/1\n"))
        if load_if == 0:
            self.gen = Generator(
                num_classes=num_classes, gen_mode=True, backbone=backbone['gen'], feature_nums=feature_nums
            )
            self.dis = Discriminator(
                num_classes=num_classes, gen_mode=False, backbone=backbone['dis'], feature_nums=feature_nums
            )
        else:
            self.gen = torch.load('./save/epoch-4987-gen.pth')
            self.dis = torch.load('./save/epoch-4987-dis.pth')
        self.gen_optimizer = torch.optim.Adam(params=self.gen.parameters(), lr=float(gen_lr), betas=(0.9, 0.999))
        self.dis_optimizer = torch.optim.Adam(params=self.dis.parameters(), lr=float(gen_lr), betas=(0.9, 0.999))

        if torch.cuda.is_available():
            self.gen, self.dis = self.gen.cuda(), self.dis.cuda()

        # self.gen_optimizer = torch.optim.Adam(
        #     lr=float(gen_lr), weight_decay=float(gen_wd), params=self.gen.parameters(), betas=gen_betas
        # )
        # self.gen_optimizer = torch.optim.Adam([
        #     {'params': self.dis.dis_adversarial.parameters(), 'lr': float(gen_lr)},
        #     {'params': self.dis.dis_classifier.parameters(), 'lr': 1e-3},
        # ], betas=gen_betas, weight_decay=float(gen_wd))

        self.dis_optimizer = torch.optim.Adam(
            lr=float(dis_lr['gan']), weight_decay=float(dis_wd), params=self.dis.parameters(), betas=dis_betas
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.gen_optimizer,
                                                              milestones=[30, 100, 500, 4000], gamma=0.1)
        # self.dis_optimizer_adversarial = torch.optim.Adam(
        #     lr=float(dis_lr['gan']), weight_decay=float(dis_wd), params=self.dis.dis_adversarial.parameters(), betas=dis_betas
        # )
        #
        # self.dis_optimizer_classifier = torch.optim.SGD(
        #     lr=float(dis_lr['cls']), weight_decay=float(dis_wd), params=self.dis.dis_classifier.parameters(), momentum=0.9
        # )

        self.train_loader, self.valid_loader, self.test_loader = get_loader()
        self.epoch, self.seed = int(train_configs['num_epochs']), int(train_configs['seed'])
        self.configs = configs
        self.best_acc, self.num_classes = 0.0, num_classes
        self.dis_loss, self.gen_loss = Dis_loss(), Gen_loss()
        if torch.cuda.is_available():
            self.dis_loss, self.gen_loss = self.dis_loss.cuda(), self.gen_loss.cuda()
        self.flag = False

    def gen_update(self, real_data,  label, randn_input):
        self.dis.eval()
        self.gen.train()
        self.gen_optimizer.zero_grad()
        gen_loss_value, loss_dict = self.gen_loss(real_data, label, randn_input, self.dis, self.gen)
        gen_loss_value.backward()
        self.gen_optimizer.step()
        return loss_dict

    def dis_update(self, real_data, label, randn_input):
        self.dis_optimizer.zero_grad()
        # self.dis_optimizer_adversarial.zero_grad()
        # self.dis_optimizer_classifier.zero_grad()

        self.gen.eval()
        self.dis.train()

        # dis_gan_loss, dis_cls_loss, loss_dict = self.dis_loss(real_data, label, randn_input, self.dis, self.gen, self.lambda_gp)
        total_loss, loss_dict = self.dis_loss(real_data, label, randn_input, self.dis, self.gen,
                                              self.lambda_gp)
        total_loss.backward()
        self.dis_optimizer.step()
        # self.dis_optimizer_adversarial.step()
        # dis_cls_loss.backward()
        # self.dis_optimizer_classifier.step()
        return loss_dict

    def train(self):
        for epoch in range(0, self.epoch):
            self.dis.train()
            self.gen.train()
            for batch_cnt, (data, label) in enumerate(self.train_loader):
                randn_input = torch.randn(data.size(0), data.size(1), data.size(2))
                if torch.cuda.is_available():
                    data, label, randn_input = data.cuda(), label.cuda(), randn_input.cuda()
                dis_loss_dict = self.dis_update(data, label, randn_input)
                # gen_loss_dict = self.gen_update(label, randn_input)
                if batch_cnt % 1 == 0:
                    gen_loss_dict = self.gen_update(data, label, randn_input)
            # self.scheduler.step(epoch=epoch)

            print(f"--------- epoch = {epoch}, batch_cnt = {batch_cnt} ---------")
            print(f"Generator loss: total loss = {gen_loss_dict['total_loss']} / "
            f"Adversarial loss = {gen_loss_dict['gan_loss'] } / "
            f"classification_loss = {gen_loss_dict['cls_loss_fake']} / "
            f"norm_loss={gen_loss_dict['norm_loss']} ")

            print(f"discriminator loss: total loss = {dis_loss_dict['total_loss']} / "
            f"Adversarial loss = {dis_loss_dict['gan_loss']} / "
            f"real sample classification loss = {dis_loss_dict['cls_loss_real']} / ")
            # f "Gradient Penalty regular term = {dis_loss_dict['gradient_penalty']} ")
            self.evaluate(epoch=epoch, data_type='valid')
            if self.flag:
                break

    def evaluate(self, epoch, data_type):
        self.gen.eval()
        self.dis.eval()
        if data_type == "valid":
            loader = self.valid_loader
        elif data_type == "test":
            loader = self.test_loader
        data_num = loader.dataset.len
        correct = 0
        data_list, label_list = [],[]

        with torch.no_grad():
            for batch_cnt, (data, label) in enumerate(loader):
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                fake_data = self.gen(data, label)
                out, cls_logits = self.dis(fake_data, label)
                _, predicted = torch.max(cls_logits, dim=1)
                correct += (predicted == label).sum().item()

                data_transform = fake_data.cpu().detach().numpy().tolist()
                data_list.append(data_transform)
                label_list.append(label.cpu().detach().numpy().tolist())
        params = json_loader('./save/stat_ori_data.json')
        data_list = np.array(inverse(data_list, self.configs))
        mean_data, std_data, max_data, min_data = np.mean(data_list), np.std(data_list),np.max(data_list), np.min(data_list)
        ori_mean, ori_std, ori_max, ori_min = params['mean'], params['std'],params['max'], params['min']
        eval_acc = correct / data_num * 100.0

        num_indicator = data_list[0][0][0][0]
        # print("num_indicator:", num_indicator)
        self.best_acc = max(self.best_acc, eval_acc)
        if self.best_acc > 99.5:
            print(" ---------------- ")
            if not self.flag:
                if np.abs(mean_data - ori_mean) < 10 and np.abs(std_data - ori_std) < 10 and\
                        49.0 < num_indicator < 52.0:
                #         and np.abs(max_data - ori_max) < 200 and np.abs(min_data - ori_min) < 1.0:
                    gen_model_name = "epoch-" + str(epoch) + "-acc-" + str(round(self.best_acc, 2)) + "-gen.pth"
                    dis_model_name = "epoch-" + str(epoch) + "-acc-" + str(round(self.best_acc, 2)) + "-dis.pth"
                    gen_model_path = os.path.join('./save/model/gen', gen_model_name)
                    dis_model_path = os.path.join('./save/model/dis', dis_model_name)
                    torch.save(self.gen, gen_model_path)
                    torch.save(self.dis, dis_model_path)
                    self.flag = True

        print(f"epoch = {epoch}, Optimal Classification Results for Pseudo Data Discriminators： acc = {round(self.best_acc, 5)} %, "
              f"mean = {np.mean(data_list)}, std = {np.std(data_list)}"
              f"max = {np.max(data_list)}, min = {np.min(data_list)}")
        print(f"标签{int(label_list[0][0])}fake data：")
        print(data_list[0][0])

        # return round(eval_acc, 2)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.configs["train"]["num_gpu"] > 0:
            torch.cuda.manual_seed_all(self.seed)

    def run(self):
        self.set_seed()
        self.train()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
