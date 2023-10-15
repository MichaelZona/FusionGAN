

class LCLoss():

	def contentFunc(self):
		# conv_3_3_layer = 14
		c = nn.Conv2d(1, 1, 3, stride=1, bias=False)
		c.weight.data = torch.tensor([[[[1.0, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]]])
		model = nn.Sequential(
			c
		)
		model = model.cuda()
		return model

	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
		for param in self.contentFunc.parameters():
			param.requires_grad = False

	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss