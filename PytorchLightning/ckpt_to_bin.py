from PytorchLightning.model import BaselineModel

if __name__ == '__main__':

    path = 'lightning_logs/gpt/checkpoints/'
    model = BaselineModel.load_from_checkpoint(path + 'epoch=5-step=7632.ckpt')
    model.model.save_pretrained(path)

