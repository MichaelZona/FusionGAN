import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import torch.nn.modules.loss as Loss
from complex_net.cmplx_blocks import batch_norm
from configs import config

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def __init__(self, opt):
		super(ConditionalGAN, self).__init__(opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.pre_A = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.post_A = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.needCompareK = "k_stack_self_uresnet" in opt.which_model_netG
		self.nok = "no_k" in opt.which_model_netG
		self.nos = "no_s" in opt.which_model_netG and "no_stack" not in opt.which_model_netG
		self.isDeblurGAN = opt.which_model_netG == "resnet_9blocks"

		# load/define networks
		# Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = not opt.gan_type == 'wgan-gp'
		print("Use Parallel = ", "True" if use_parallel else "False")
		self.netG = networks.define_G(
			opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
			not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual
		)
		if self.isTrain:
			use_sigmoid = opt.gan_type == 'gan'
			self.netD = networks.define_D(
				opt.output_nc, opt.ndf, opt.which_model_netD,
				opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel
			)
		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam( self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
			self.optimizer_D = torch.optim.Adam( self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
												
			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.contentLoss, self.VGGLoss, self.LCLoss, self.GFLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netD)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		inputA = input['A' if AtoB else 'B']
		inputB = input['B' if AtoB else 'A']
		self.input_A.resize_(inputA.size()).copy_(inputA)
		self.input_B.resize_(inputB.size()).copy_(inputB)
		self.no_pre_post = False
		try:
			preA = input['pre_A']
			self.pre_A = preA.resize_(preA.size()).copy_(preA)
			postA = input['post_A']
			self.post_A = postA.resize_(postA.size()).copy_(postA)
		except KeyError:
			self.no_pre_post = True
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		if self.no_pre_post:
			self.fake_B = self.netG.forward(self.real_A)
			if self.needCompareK:
				if not self.nok:
					self.KData = self.netG.forward(self.real_A, getKPredict=True)
				if not self.nos:
					self.SData = self.netG.forward(self.real_A, getSPredict=True)
		else:
			self.pre_A = Variable(self.pre_A)
			self.post_A = Variable(self.post_A)
			self.fake_B = self.netG.forward(self.real_A,self.pre_A,self.post_A)
			if self.needCompareK:
				if not self.nok:
					self.KData = self.netG.forward(self.real_A,self.pre_A,self.post_A, getKPredict=True)
				if not self.nos:
					self.SData = self.netG.forward(self.real_A,self.pre_A,self.post_A, getSPredict=True)
		self.real_B = Variable(self.input_B)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		if self.no_pre_post:
			self.fake_B = self.netG.forward(self.real_A)
		else:
			self.pre_A = Variable(self.pre_A)
			self.post_A = Variable(self.post_A)
			self.fake_B = self.netG.forward(self.real_A,self.pre_A,self.post_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B)

		self.loss_D.backward(retain_graph=True)

	def backward_G(self, opt=None):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B) * self.opt.adv ##adver loss
		# Second, G(A) = B
		self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.content
		self.loss_G_VGG = self.VGGLoss.get_loss(self.fake_B, self.real_B) * self.opt.vgg ##perpectual loss
		self.loss_G_LC = self.LCLoss.get_loss(self.fake_B, self.real_B) * self.opt.lc ##local consistency loss
		if self.isDeblurGAN:
			self.loss_G_LC = 0

		if self.needCompareK:
			if not self.nok:
				self.loss_G_Gf = self.GFLoss.get_loss(self.KData['k'], torch.fft.fftshift(torch.fft.fft2(self.real_B))) * self.opt.gf
				self.loss_G_DCf = self.contentLoss.get_loss(self.KData['s'], self.real_B) * self.opt.gs
				self.loss_G_f = self.loss_G_Gf + self.loss_G_DCf

			if not self.nos:
				self.loss_G_Gs = self.contentLoss.get_loss(self.SData['s'], self.real_B) * self.opt.gs
				self.loss_G_DCs = self.GFLoss.get_loss(self.SData['k'], torch.fft.fftshift(torch.fft.fft2(self.real_B))) * self.opt.gf
				self.loss_G_s = self.loss_G_Gs + self.loss_G_DCs
		else:
			self.loss_G_f = 0
			self.loss_G_s = 0

		if self.nok:
			self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_G_VGG + self.loss_G_LC + self.loss_G_s
		elif self.nos:
			self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_G_VGG + self.loss_G_LC + self.loss_G_f
		else:
			self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.loss_G_VGG + self.loss_G_LC + self.loss_G_s +self.loss_G_f

		# if self.opt.which_model_netG == 'knet':
		# 	radial_normalizer = batch_norm(
		# 		in_channels=config.in_channels,
		# 	).cuda()
		# 	# k_x = torch.fft.fft2(self.real_B)
		# 	# real = torch.unsqueeze(torch.real(k_x), -1)
		# 	# imag = torch.unsqueeze(torch.imag(k_x), -1)
		# 	# k_x = torch.cat((real, imag), -1)
		# 	# y = radial_normalizer(k_x)
		# 	# k_real, k_imag = torch.unbind(y, dim=-1)
		# 	# k_x = torch.complex(k_real, k_imag)
		# 	loss = Loss.MSELoss()
		# 	self.loss_G = loss(self.fake_B, self.real_B) * self.opt.adv

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
							('G_L1', self.loss_G_Content.item()),
							('G_VGG', self.loss_G_VGG.item()),
							('G_LC', self.loss_G_LC.item() if not self.isDeblurGAN else 0),
							('G_GF', self.loss_G_Gf.item() if self.needCompareK and not self.nok else 0),
							('G_DCF', self.loss_G_DCf.item() if self.needCompareK and not self.nok else 0),
							('G_f', self.loss_G_f.item() if self.needCompareK and not self.nok else 0),
							('G_GS', self.loss_G_Gs.item() if self.needCompareK and not self.nos else 0),
							('G_DCS', self.loss_G_DCs.item() if self.needCompareK and not self.nos else 0),
							('G_s', self.loss_G_s.item() if self.needCompareK and not self.nos else 0),
							('D_real+fake', self.loss_D.item())
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.repeat(1,3,1,1).data)
		fake_B = util.tensor2im(self.fake_B.repeat(1,3,1,1).data)
		real_B = util.tensor2im(self.real_B.repeat(1,3,1,1).data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
