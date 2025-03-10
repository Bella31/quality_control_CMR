import logging
import os
from sacred.observers import FileStorageObserver
from sacred import Experiment
import torch
from utils.CA import AE, plot_history
from utils.utils_fetal import FetalDataLoader, list_load, transform, transform_augmentation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ex = Experiment()

@ex.config
def config():
    my_path = os.path.abspath(os.path.dirname(__file__))
    data_path = '/home/bella/Phd/data/body/TRUFI/TRUFI'
    train_valid_lists_path = '/home/bella/Phd/code/code_bella/fetal_mr/config/config_body/quality_estimation/trufi_qe/0'
    data_size = [512, 512]
    num_epochs = 100
    opt_params = {
        "BATCH_SIZE": 32,
        "DA": False,
        "latent_size": 100,
        "optimizer": torch.optim.Adam,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "functions": ["BKGDLoss", "BKMSELoss"],
        "settling_epochs_BKGDLoss": 10,
        "settling_epochs_BKMSELoss": 0
    }


@ex.main
def my_main(data_path, train_valid_lists_path, num_epochs, opt_params):
  #  train_masks, validation_masks = Preprocess.load_preprocess_data(data_path, train_valid_lists_path, data_size)
  ae = AE(**opt_params).to(device)

  ckpt = None
  if ckpt is not None:
      ckpt = torch.load(ckpt)
      ae.load_state_dict(ckpt["AE"])
      ae.optimizer.load_state_dict(ckpt["AE_optim"])
      start = ckpt["epoch"] + 1
  else:
      start = 0

  print(ae)

  training_file = os.path.join(train_valid_lists_path, 'training_ids.txt')
  validation_file = os.path.join(train_valid_lists_path, 'validation_ids.txt')
  train_ids = list_load(training_file)
  val_ids = list_load(validation_file)

  plot_history(
      ae.training_routine(
          range(start, num_epochs),
          FetalDataLoader(data_path, patient_ids=train_ids, batch_size=opt_params["BATCH_SIZE"], filename='truth.nii.gz',
                          transform=transform_augmentation if opt_params["DA"] else transform),
          FetalDataLoader(data_path, patient_ids=val_ids, batch_size=opt_params["BATCH_SIZE"], filename='truth.nii.gz',
                          transform=transform),
          os.path.join(ex.observers[0].dir, "checkpoints")
      )
    )


if __name__ == '__main__':

    log_dir = '../../log/'
    log_level = logging.INFO
    my_path = os.path.abspath(os.path.dirname(__file__))
    log_path = os.path.join(my_path, log_dir)

    # uid = uuid.uuid4().hex
    # fs_observer = FileStorageObserver.create(os.path.join(log_path, uid))
    fs_observer = FileStorageObserver.create(log_path)

    ex.observers.append(fs_observer)

    # initialize logger
    logger = logging.getLogger()
    hdlr = logging.FileHandler(os.path.join(ex.observers[0].basedir, 'messages.log'))
    FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(filename)20s:%(lineno)-5s %(funcName)-25s %(message)s"
    formatter = logging.Formatter(FORMAT)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    # logger.removeHandler(lhStdout)
    logger.setLevel(log_level)
    ex.logger = logger
    logging.info('Experiment {}, run {} initialized'.format(ex.path, ex.current_run))

    ex.run_commandline()