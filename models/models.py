from .conditional_gan_model import ConditionalGAN

def create_model(opt):
	model = None
	if opt.model == 'test':
		from .test_model import TestModel
		model = TestModel( opt )
	else:
		model = ConditionalGAN(opt)
	# model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
