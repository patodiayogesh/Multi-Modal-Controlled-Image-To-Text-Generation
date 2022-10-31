import copy

class Trainer:

    def __init__(self,
                 model,
                 dataset,
                 epochs=20,
                 patience=2):

        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_path = 'checkpoints/'
        self.dataset.set_model_variables(self.model)

    def fit(self):

        prev_loss = float('inf')
        patience = 0
        best_model = None
        best_epoch = -1
        self.dataset.setup('fit')

        for epoch in range(self.epochs):

            train_loss = self.model.train(epoch,
                                          self.dataset.train_dataloader())
            val_loss = self.model.test(epoch,
                                       self.dataset.val_dataloader())
            self.model.lr_scheduler.step()

            if val_loss >= prev_loss:
                if patience >= self.patience:
                    break
                else:
                    patience += 1
            else:
                patience = 0
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch

            self.model.save_pretrained(f'{self.checkpoint_path}epoch_{epoch}/')

        best_model.model_save_pretrained(f'{self.checkpoint_path}best_epoch_{best_epoch}/')

    def inference(self):

        self.dataset.setup('predict')
        dataloader = self.dataset.predict_dataloader()
        self.model.predict(dataloader)


