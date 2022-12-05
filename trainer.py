import copy
import os

class Trainer:

    def __init__(self,
                 model,
                 dataset,
                 epochs=20,
                 patience=3):

        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_path = 'checkpoints_vqa/'
        self.version = self._get_run_version()
        self.checkpoint_path = f'{self.checkpoint_path}{self.version}/'
        self.dataset.set_model_variables(self.model)

    def fit(self):

        prev_loss = float('inf')
        patience = 0
        best_model = None
        best_epoch = -1
        self.dataset.setup('fit')

        for epoch in range(self.epochs):

            self.dataset.epoch = epoch

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
            prev_loss = val_loss

            self.model.save_pretrained(f'{self.checkpoint_path}epoch_{epoch}/')

        best_model.save_pretrained(f'{self.checkpoint_path}best_epoch_{best_epoch}/')

    def inference(self):

        self.dataset.setup('predict')
        dataloader = self.dataset.predict_dataloader()
        self.model.predict(dataloader,
                           filename=self.dataset.predict_file)

    def _get_run_version(self):
        '''
        Function to check last version no and return new dir to save model
        '''

        folder = 'checkpoints/'
        sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        versions = []
        for x in sub_folders:
            if x.startswith('version'):
                versions.append(x)
        if versions == []:
            return 'version_0'
        versions.sort(reverse=True)
        last_ver = int(versions[0].split('_')[1])
        new_ver = 'version_' + str(int(last_ver)+1)
        return new_ver
