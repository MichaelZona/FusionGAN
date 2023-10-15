import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.pre_A = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.post_A = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']
        input_B = input['B']
        temp = self.input_B.clone()
        temp.resize_(input_B.size()).copy_(input_B)
        self.input_B = temp

        self.no_pre_post = False
        try:
            preA = input['pre_A']
            self.pre_A = preA.resize_(preA.size()).copy_(preA)
            postA = input['post_A']
            self.post_A = postA.resize_(postA.size()).copy_(postA)
        except KeyError:
            self.no_pre_post = True

    def test(self):
        with torch.no_grad():
            if self.no_pre_post:
                self.real_A = Variable(self.input_A)
                self.fake_B = self.netG.forward(self.real_A)
                self.real_B = Variable(self.input_B)
            else:
                self.real_A = Variable(self.input_A)
                self.pre_A = Variable(self.pre_A)
                self.post_A = Variable(self.post_A)
                self.fake_B = self.netG.forward(self.real_A,self.pre_A,self.post_A)
                self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.repeat(1,3,1,1).data)
        fake_B = util.tensor2im(self.fake_B.repeat(1,3,1,1).data)
        real_B = util.tensor2im(self.real_B.repeat(1,3,1,1).data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),('real_B',real_B)])
